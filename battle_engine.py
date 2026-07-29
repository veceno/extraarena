"""
Класс-адаптер BattleEngine для совместимости со старым кодом.
Связывает старый API с новой архитектурой core/.
"""

from __future__ import annotations
import asyncio
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Dict, List
from uuid import UUID, uuid4
import logging
import re
import time

from core.actions import AttackAction, BaseAction, EndTurnAction, ManaDrawAction, PlayCardAction
from core.classic_setup import create_classic_game_state
from core.engine import ArenaEnvironment
from core.effects import get_taunt_targets, has_taunt
from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState as CorePlayerState, ReplacementStatus
from core.v5_dataset import (
    ACTION_SOURCES as V5_ACTION_SOURCES,
    V5DatasetRecorder,
    canonical_actor_type as canonical_v5_actor_type,
)
from core.nemesis_dataset import NemesisBattleCollector
from infrastructure.card_assets import card_asset_url
from infrastructure.match_modes import ModeConfig, resolve_mode_config, serialize_mode_config


# Глобальный словарь активных матчей
ACTIVE_MATCHES: Dict[str, "BattleEngine"] = {}

TRAIN_V3_CARD_PARAMS_SCHEMA = "train_v3_card_params_v1"
TRAIN_V3_ACTION_CONTEXT_SCHEMA = "train_v3_action_context_v1"
TRAIN_V3_DECK_PARAMS_SCHEMA = "train_v3_deck_params_v1"

TALKIE_CATALOG: dict[str, dict[str, str]] = {
    "1": {"sound": "neutral"},
    "2": {"sound": "sad"},
    "3": {"sound": "rude"},
    "4": {"sound": "rude"},
    "5": {"sound": "happy"},
    "6": {"sound": "neutral"},
    "7": {"sound": "happy"},
}

TALKIE_TIER_LIMITS: dict[str, dict[str, float | int | None]] = {
    "inactive": {"max_per_turn": 1, "cooldown_seconds": None},
    "active": {"max_per_turn": 2, "cooldown_seconds": 1.0},
    "ultra": {"max_per_turn": 3, "cooldown_seconds": 1.0},
}


class BattleEventEmitter:
    """Система событий для рассылки обновлений клиентам."""
    
    def __init__(self) -> None:
        self._listeners: Dict[str, List[Any]] = {}
    
    def on(self, event_type: str, callback: Any) -> None:
        """Регистрирует обработчик события."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
    
    def emit(self, event_type: str, match_id: str, data: dict[str, Any]) -> None:
        """Генерирует событие."""
        if event_type not in self._listeners:
            return
        for callback in self._listeners[event_type]:
            try:
                callback(match_id, data)
            except Exception as exc:
                logging.getLogger(__name__).warning("Event callback failed: %s", exc)


class BattleEngine:
    """
    Адаптер между старым API и core/engine.ArenaEnvironment.
    Управляет жизненным циклом матча и сериализацией состояния для фронтенда.
    """
    
    def __init__(
        self,
        db: Any = None,
        match_id: Any = None,
        player_ids: Optional[list[int]] = None,
        is_bot_match: bool = False,
        p1_deck_ids: Optional[List[str]] = None,
        p2_deck_ids: Optional[List[str]] = None,
        active_matches: Optional[Dict[str, "BattleEngine"]] = None,
        p1_hero_hp_override: Optional[int] = None,
        p2_hero_hp_override: Optional[int] = None,
        card_cache: Optional[Dict[str, Any]] = None,
        event_emitter: Optional[BattleEventEmitter] = None,
        p1_name: Optional[str] = None,
        p1_avatar_url: Optional[str] = None,
        p2_name: Optional[str] = None,
        p2_avatar_url: Optional[str] = None,
        game_mode: str = "classic",
    ) -> None:
        """Инициализация боевого движка."""
        self._db = db
        self.match_id = match_id
        # ``match_id`` is a gameplay identifier and can be reused when an
        # in-memory friendly match is reconstructed after a process restart.
        # The dataset journal, however, seals battle IDs immutably.  Keep a
        # separate per-engine trace generation so a rehydrated engine cannot
        # collide with the sealed prefix from the previous process.
        self.v5_dataset_trace_id = str(match_id or "")
        self.v5_dataset_generation = 1
        self.v5_dataset_generation_reason = "initial"
        self.player_ids = player_ids or []
        self.is_bot_match = is_bot_match
        self.bot_id: Optional[int] = None
        self.bot_difficulty: str = "lite"  # Inference profile key.
        self.bot_difficulty_label: str = "lite"
        self.bot_strength_tier: str = "lite"
        self.bot_brain_profile: Optional[str] = None
        # Runtime dependencies are bound by web.server to the aiohttp app that
        # created this match.  They deliberately live on the match instead of
        # being resolved from process globals: two app generations may overlap
        # briefly during a graceful reload, and an old match must never switch
        # to the new app's ONNX Runtime sessions.
        self.runtime_owner_app: Any = None
        self.berserk_brain: Any = None
        self.extra_lr_aux_runtime: Any = None
        self._active_matches = active_matches if active_matches is not None else ACTIVE_MATCHES
        self._event_emitter = event_emitter
        self.current_player_id: Optional[int] = None
        self.turn = 0
        self.is_ended = False
        self.game_over_processed = False
        self.rewards_granted = False
        self.turn_start_time: Optional[float] = None
        self.turn_start_monotonic: Optional[float] = None
        self.match_start_time: Optional[float] = None
        self.match_start_monotonic: Optional[float] = None
        self.mode_config: ModeConfig = resolve_mode_config(game_mode)
        self.game_mode = self.mode_config.mode_id
        self.ruleset = self.mode_config.ruleset
        self.turn_duration = self.mode_config.classic.turn_duration_seconds
        self.turn_time_history: list[dict[str, Any]] = []
        self.client_ready = False
        self.client_ready_users: set[int] = set()
        self._card_cache: Dict[str, Any] = card_cache or {}
        self._logger = logging.getLogger(__name__)
        
        # Счётчики таймаутов для AFK-детекции
        self.p1_consecutive_timeouts: int = 0
        self.p2_consecutive_timeouts: int = 0
        
        # Ядро боя — ArenaEnvironment из core/
        self._arena: Optional[ArenaEnvironment] = None
        
        # Метаданные игроков для UI
        self._p1_name = p1_name or "Игрок 1"
        self._p1_avatar_url = p1_avatar_url
        self._p2_name = p2_name or "Игрок 2"
        self._p2_avatar_url = p2_avatar_url
        self._p1_trophies: int = 0
        self._p2_trophies: int = 0
        self._p1_clan: str = ""
        self._p2_clan: str = ""
        self._p1_title: str = ""
        self._p1_rarity: str = ""
        self._p2_title: str = ""
        self._p2_rarity: str = ""
        self._p1_extra_pass: Optional[str] = None
        self._p1_background_url: Optional[str] = None
        self._p1_nickname_glow_disabled: bool = False
        self._p1_hide_player_id_public: bool = False
        self._p2_extra_pass: Optional[str] = None
        self._p2_background_url: Optional[str] = None
        self._p2_nickname_glow_disabled: bool = False
        self._p2_hide_player_id_public: bool = False
        
        # Для совместимости со старым кодом (минимальные заглушки)
        p1_id = player_ids[0] if player_ids and len(player_ids) > 0 else 1
        p2_id = player_ids[1] if player_ids and len(player_ids) > 1 else 2
        self._p1_id = p1_id
        self._p2_id = p2_id
        self._talkie_enabled_by_user: dict[int, bool] = {}
        self._talkie_usage_by_turn: dict[tuple[int, int], list[float]] = {}
        self._talkie_event_seq: int = 0

        # Analytics data collection
        self._analytics_actions: list[dict[str, Any]] = []
        self._analytics_flushed: bool = False
        self._p1_initial_deck_ids: list[int] = []
        self._p2_initial_deck_ids: list[int] = []
        self._p1_initial_deck_params: dict[str, Any] = {}
        self._p2_initial_deck_params: dict[str, Any] = {}
        # Private omniscient Phase-C/auxiliary-model recorder.  It is
        # deliberately match-owned and in-memory: server/DB code checkpoints a
        # detached snapshot through the public methods below.
        self.v5_dataset_recorder = V5DatasetRecorder()
        self.nemesis_collector: Optional[NemesisBattleCollector] = None

    @staticmethod
    @lru_cache(maxsize=1)
    def _nemesis_catalog_hash() -> Optional[str]:
        path = Path(__file__).resolve().parent / "ai" / "cards.json"
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None

    @staticmethod
    def _effective_deck_manifest(
        deck: list[CardInstance],
    ) -> list[dict[str, Any]]:
        """Record the levels actually applied by the engine, not requested levels."""

        return [
            {
                "card_id": int(card.card_id),
                "level": int(card.level),
                "instance_id": None,
                "slot": slot,
            }
            for slot, card in enumerate(deck)
        ]

    # =========================================================================
    # PRIVATE V5 DATASET LIFECYCLE
    # =========================================================================

    def start_new_v5_dataset_generation(
        self,
        *,
        reason: str,
        trace_id: Optional[str] = None,
    ) -> str:
        """Assign a fresh storage ID before a rehydrated match starts recording."""

        if bool(getattr(self.v5_dataset_recorder, "_started", False)):
            raise RuntimeError("cannot rotate V5 dataset trace after recording started")
        gameplay_match_id = str(self.match_id or "")
        if not gameplay_match_id:
            raise ValueError("match_id is required before rotating V5 dataset trace")
        next_generation = max(1, int(self.v5_dataset_generation)) + 1
        generation_reason = str(reason or "rehydrate")
        candidate = str(trace_id or "").strip()
        if not candidate:
            candidate = f"v5g{next_generation}-{uuid4().hex}"
        if len(candidate) > 128 or re.fullmatch(r"[A-Za-z0-9._-]+", candidate) is None:
            raise ValueError("V5 dataset trace_id must be a safe <=128 character component")
        if candidate == gameplay_match_id:
            raise ValueError("rehydrated V5 dataset trace must differ from match_id")
        self.v5_dataset_generation = next_generation
        self.v5_dataset_generation_reason = generation_reason
        self.v5_dataset_trace_id = candidate
        return candidate

    @staticmethod
    def _selection_deck_manifest(
        deck_ids: list[Any],
        levels: dict[int, int],
        slot_levels: Optional[list[int]] = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for slot, raw_card_id in enumerate(deck_ids):
            try:
                card_id = int(str(raw_card_id).split(":", 1)[0])
            except (TypeError, ValueError):
                continue
            if slot_levels is not None and slot < len(slot_levels):
                level = int(slot_levels[slot])
            else:
                level = int(levels.get(card_id, 1) or 1)
            rows.append(
                {
                    "card_id": card_id,
                    "level": level,
                    "instance_id": None,
                    "slot": slot,
                }
            )
        return rows

    def _start_v5_dataset(
        self,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self._arena:
            return
        base_metadata: dict[str, Any] = {
            "game_mode": self.game_mode,
            "ruleset": self.ruleset,
            "mode_config": serialize_mode_config(self.mode_config),
            "bot_policy": {
                "profile": self.bot_brain_profile,
                "difficulty": self.bot_difficulty,
                "difficulty_label": self.bot_difficulty_label,
                "strength_tier": self.bot_strength_tier,
            },
        }
        if metadata:
            base_metadata.update(metadata)
        self.v5_dataset_recorder.start_match(self, metadata=base_metadata)

    def set_v5_dataset_metadata(self, metadata: dict[str, Any]) -> None:
        """Merge model/deck provenance into the private recorder metadata."""

        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")
        if self._arena:
            self._start_v5_dataset()
            self.v5_dataset_recorder.merge_metadata(self, metadata)

    def mark_v5_policy_degraded(self, code: str) -> str:
        """Record one stable policy-failure code without exposing exception text."""

        if not self._arena:
            raise RuntimeError("arena_not_initialized")
        self._start_v5_dataset()
        return self.v5_dataset_recorder.mark_policy_degraded(self, code)

    def mark_v5_battle_started(
        self,
        *,
        reason: str = "client_ready",
        now_monotonic: Optional[float] = None,
        now_wall: Optional[float] = None,
    ) -> None:
        """Anchor TimeStamp duration to the first observable battle state."""

        if not self._arena:
            return
        self._start_v5_dataset()
        self.v5_dataset_recorder.mark_battle_started(
            self,
            reason=reason,
            now_monotonic=now_monotonic,
            now_wall=now_wall,
        )

    def arm_human_decision_clock(
        self,
        user_id: int,
        *,
        now_monotonic: Optional[float] = None,
        censored: bool = False,
        censor_reason: Optional[str] = None,
    ) -> None:
        """Start a monotonic decision interval after private state delivery."""

        self.v5_dataset_recorder.arm_human_decision_clock(
            int(user_id),
            now_monotonic=now_monotonic,
            censored=censored,
            censor_reason=censor_reason,
        )

    def censor_human_decision_clock(self, user_id: int, reason: str) -> None:
        """Mark the next human timing label unusable without discarding action."""

        self.v5_dataset_recorder.censor_human_decision_clock(
            int(user_id), str(reason)
        )

    def _classify_v5_action_source(
        self,
        user_id: int,
        action: Optional[BaseAction] = None,
        *,
        explicit: Optional[str] = None,
    ) -> str:
        if explicit is not None:
            source = str(explicit).lower()
            if source in V5_ACTION_SOURCES:
                return source
        if isinstance(action, EndTurnAction):
            try:
                if self.is_turn_expired():
                    return "timeout"
            except Exception:  # noqa: BLE001
                pass
        if self._arena:
            status = self.get_player_replacement_status(int(user_id))
            if status in {ReplacementStatus.AFK, ReplacementStatus.SURRENDERED}:
                return "replacement_bot"
            if self.is_bot(int(user_id)):
                return "bot"
        return "human"

    def capture_action_context(
        self,
        user_id: int,
        action_json: dict[str, Any],
        *,
        decision_source: Optional[str] = None,
        control_source: Optional[str] = None,
        actor_type: Optional[str] = None,
        request_monotonic: Optional[float] = None,
        client_action_id: Optional[str] = None,
        human_decision_time_ms: Optional[float] = None,
        decision_time_censored: Optional[bool] = None,
        decision_censor_reason: Optional[str] = None,
        metronome_prediction_ms: Optional[float] = None,
        metronome_applied_ms: Optional[float] = None,
        metronome_fallback_used: Optional[bool] = None,
        quality_score: Optional[float] = None,
        legacy_action_index: Optional[int] = None,
        extra_context: Optional[dict[str, Any]] = None,
        queue: bool = True,
    ) -> dict[str, Any]:
        """Capture one request/decision context for the next engine mutation."""

        inferred_action: Optional[BaseAction] = None
        if str(action_json.get("type") or "") == "end_turn":
            inferred_action = EndTurnAction()
        source = self._classify_v5_action_source(
            int(user_id), inferred_action, explicit=decision_source
        )
        return self.v5_dataset_recorder.capture_action_context(
            user_id=int(user_id),
            action_json=action_json,
            decision_source=source,
            control_source=control_source,
            actor_type=actor_type or canonical_v5_actor_type(source),
            request_monotonic=request_monotonic,
            client_action_id=client_action_id,
            human_decision_time_ms=human_decision_time_ms,
            decision_time_censored=decision_time_censored,
            decision_censor_reason=decision_censor_reason,
            metronome_prediction_ms=metronome_prediction_ms,
            metronome_applied_ms=metronome_applied_ms,
            metronome_fallback_used=metronome_fallback_used,
            quality_score=quality_score,
            legacy_action_index=legacy_action_index,
            extra_context=extra_context,
            queue=queue,
        )

    def get_v5_dataset_snapshot(self) -> dict[str, Any]:
        """Return a detached, private copy of the current match dataset."""

        if self._arena:
            self._start_v5_dataset()
        return self.v5_dataset_recorder.snapshot(self if self._arena else None)

    def checkpoint_v5_dataset(
        self,
        *,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a detached storage checkpoint for a DB/outbox writer."""

        if not self._arena:
            return self.v5_dataset_recorder.snapshot()
        self._start_v5_dataset()
        return self.v5_dataset_recorder.checkpoint(self, reason=reason)

    def finalize_v5_dataset(
        self,
        *,
        winner_user_id: Optional[int] = None,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Finalize once and return an idempotent detached terminal snapshot."""

        if not self._arena:
            return self.v5_dataset_recorder.snapshot()
        self._start_v5_dataset()
        if self.nemesis_collector is not None and status:
            try:
                normalized_status = str(getattr(status, "value", status))
                duration_seconds = None
                if self.match_start_monotonic is not None:
                    duration_seconds = max(
                        0,
                        int(round(time.monotonic() - self.match_start_monotonic)),
                    )
                nemesis_record = self.nemesis_collector.finalize(
                    status=normalized_status,
                    duration_seconds=duration_seconds,
                    turns_count=int(
                        getattr(self._arena.state, "turn_number", 0) or 0
                    ),
                )
                self.v5_dataset_recorder.merge_metadata(
                    self, {"nemesis_record": nemesis_record}
                )
            except Exception:
                # Dataset collection must never turn a valid terminal game into
                # a failed gameplay request. The V5 journal retains an explicit
                # failure marker for audit and the compact exporter fails closed.
                self._logger.error(
                    "Nemesis terminal seal failed: match=%s",
                    self.match_id,
                    exc_info=True,
                )
                self.v5_dataset_recorder.merge_metadata(
                    self,
                    {
                        "nemesis_collection": {
                            "status": "unavailable",
                            "reason": "terminal_seal_failed",
                        }
                    },
                )
        return self.v5_dataset_recorder.finalize(
            self,
            winner_user_id=winner_user_id,
            status=status,
            reason=reason,
            metadata=metadata,
        )

    def abort_v5_dataset(
        self,
        reason: str,
        *,
        status: str = "aborted",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Finalize a non-training terminal such as reload or disconnect."""

        if not self._arena:
            return self.v5_dataset_recorder.snapshot()
        self._start_v5_dataset()
        return self.v5_dataset_recorder.abort(
            self,
            reason=str(reason),
            status=str(status),
            metadata=metadata,
        )
    
    # =========================================================================
    # СОЗДАНИЕ МАТЧА
    # =========================================================================
    
    async def create_match(
        self,
        match_id: str,
        p1_data: Dict[str, Any],
        p2_data: Dict[str, Any],
        starting_player_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Создать матч с использованием core/engine.
        
        Args:
            match_id: ID матча
            p1_data: {user_id, deck_ids, name, avatar_url, is_bot, trophies}
            p2_data: {user_id, deck_ids, name, avatar_url, is_bot, trophies}
        """
        try:
            from core.converter import deck_from_card_ids

            if not self.mode_config.available:
                return {
                    "success": False,
                    "error": f"mode_unavailable:{self.mode_config.mode_id}",
                }
            if self.mode_config.ruleset != "classic":
                return {
                    "success": False,
                    "error": f"ruleset_not_implemented:{self.mode_config.ruleset}",
                }
            
            self._logger.info(
                "create_match: match_id=%s p1_id=%s p2_id=%s",
                match_id, p1_data.get("user_id"), p2_data.get("user_id")
            )
            
            self.match_id = match_id
            if not self.v5_dataset_trace_id:
                self.v5_dataset_trace_id = str(match_id)
            self.player_ids = [p1_data["user_id"], p2_data["user_id"]]
            self._p1_id = p1_data["user_id"]
            self._p2_id = p2_data["user_id"]
            self.is_bot_match = p1_data.get("is_bot", False) or p2_data.get("is_bot", False)

            snapshot_getter = getattr(self._db, "get_nemesis_profile_snapshot", None)

            async def _snapshot(seat: str, data: Dict[str, Any]) -> Any:
                if data.get("is_bot"):
                    return None
                if not callable(snapshot_getter):
                    return None
                try:
                    return await asyncio.wait_for(
                        snapshot_getter(
                            int(data["user_id"]),
                            history_limit=32,
                        ),
                        timeout=1.5,
                    )
                except Exception:
                    self._logger.warning(
                        "Nemesis pre-match snapshot failed: match=%s seat=%s",
                        match_id,
                        seat,
                        exc_info=True,
                    )
                    return None

            p1_nemesis_snapshot, p2_nemesis_snapshot = await asyncio.gather(
                _snapshot("p1", p1_data),
                _snapshot("p2", p2_data),
            )
            
            if p1_data.get("is_bot"):
                self.bot_id = p1_data["user_id"]
                self.bot_difficulty = p1_data.get("difficulty", "lite")
                self.bot_difficulty_label = p1_data.get("difficulty_label", self.bot_difficulty)
                self.bot_strength_tier = p1_data.get("strength_tier", self.bot_difficulty)
                self.bot_brain_profile = p1_data.get("brain_profile")
            elif p2_data.get("is_bot"):
                self.bot_id = p2_data["user_id"]
                self.bot_difficulty = p2_data.get("difficulty", "lite")
                self.bot_difficulty_label = p2_data.get("difficulty_label", self.bot_difficulty)
                self.bot_strength_tier = p2_data.get("strength_tier", self.bot_difficulty)
                self.bot_brain_profile = p2_data.get("brain_profile")
            
            # Метаданные
            self._p1_name = p1_data.get("name", "Игрок 1")
            self._p1_avatar_url = p1_data.get("avatar_url")
            self._p1_trophies = p1_data.get("trophies", 0)
            self._p1_clan = p1_data.get("clan", "")
            self._p1_title = p1_data.get("title", "")
            self._p1_rarity = p1_data.get("rarity", "")
            self._p1_extra_pass = p1_data.get("extra_pass")
            self._p1_nickname_glow_disabled = bool(p1_data.get("nickname_glow_disabled", False))
            self._p1_hide_player_id_public = bool(p1_data.get("hide_player_id_public", False))
            self._p1_background_url = p1_data.get("background_url")
            self._p2_name = p2_data.get("name", "Игрок 2")
            self._p2_avatar_url = p2_data.get("avatar_url")
            self._p2_trophies = p2_data.get("trophies", 0)
            self._p2_clan = p2_data.get("clan", "")
            self._p2_title = p2_data.get("title", "")
            self._p2_rarity = p2_data.get("rarity", "")
            self._p2_extra_pass = p2_data.get("extra_pass")
            self._p2_nickname_glow_disabled = bool(p2_data.get("nickname_glow_disabled", False))
            self._p2_hide_player_id_public = bool(p2_data.get("hide_player_id_public", False))
            self._p2_background_url = p2_data.get("background_url")
            
            # Загружаем данные карт из БД
            all_deck_ids = (p1_data.get("deck_ids") or []) + (p2_data.get("deck_ids") or [])
            cards_data = await self._load_cards_data(all_deck_ids)
            self._logger.info(
                "[ADAPTER] match=%s: загружено %d карточных записей из БД",
                self.match_id,
                len(cards_data),
            )
            
            # Загружаем уровни карт (с оверрайдом для ботов)
            p1_levels = await self._load_user_card_levels(p1_data["user_id"], p1_data.get("deck_ids") or [])
            p2_levels = await self._load_user_card_levels(p2_data["user_id"], p2_data.get("deck_ids") or [])
            p1_levels = self._apply_card_level_mode(p1_levels, p1_data.get("deck_ids") or [])
            p2_levels = self._apply_card_level_mode(p2_levels, p2_data.get("deck_ids") or [])

            # Bot card level override: slot-based per-card levels.
            # В normal режиме передаём в converter как slot_levels (per-index).
            # В disabled/max _apply_card_level_mode уже всё установила — не трогаем.
            p1_slot_levels = None
            p2_slot_levels = None
            p1_raw = p1_data.get("card_levels")
            p2_raw = p2_data.get("card_levels")
            if p1_data.get("is_bot") and p1_raw is not None and self.mode_config.classic.card_level_mode == "normal":
                p1_slot_levels = [int(l) for l in p1_raw]
                self._logger.info("[ADAPTER] Bot P1 card_levels: %s", p1_slot_levels)
            if p2_data.get("is_bot") and p2_raw is not None and self.mode_config.classic.card_level_mode == "normal":
                p2_slot_levels = [int(l) for l in p2_raw]
                self._logger.info("[ADAPTER] Bot P2 card_levels: %s", p2_slot_levels)

            # Создаем колоды CardInstance — боты через slot_levels, игроки через pX_levels
            p1_deck = deck_from_card_ids(
                p1_data.get("deck_ids") or [], cards_data, p1_levels,
                slot_levels=p1_slot_levels,
            )
            p2_deck = deck_from_card_ids(
                p2_data.get("deck_ids") or [], cards_data, p2_levels,
                slot_levels=p2_slot_levels,
            )

            # Store initial deck IDs for analytics
            self._p1_initial_deck_ids = [int(c) for c in (p1_data.get("deck_ids") or [])]
            self._p2_initial_deck_ids = [int(c) for c in (p2_data.get("deck_ids") or [])]
            self._p1_initial_deck_params = self._deck_param_snapshot(p1_deck)
            self._p2_initial_deck_params = self._deck_param_snapshot(p2_deck)

            self._logger.debug(
                "[ADAPTER] match=%s: собраны колоды (p1=%d карт, p2=%d карт)",
                self.match_id,
                len(p1_deck),
                len(p2_deck),
            )
            
            import random
            game_state = create_classic_game_state(
                p1_user_id=p1_data["user_id"],
                p2_user_id=p2_data["user_id"],
                p1_deck=p1_deck,
                p2_deck=p2_deck,
                p1_is_bot=p1_data.get("is_bot", False),
                p2_is_bot=p2_data.get("is_bot", False),
                p1_trophies=p1_data.get("trophies", 0),
                p2_trophies=p2_data.get("trophies", 0),
                starting_player_id=starting_player_id,
                rng=random.Random(),
            )

            health_multiplier = self.mode_config.classic.hero_health_multiplier
            if health_multiplier != 1.0:
                p1_hero = game_state.p1.hero
                p2_hero = game_state.p2.hero
                p1_hp = max(1, round(p1_hero.hp * health_multiplier))
                p2_hp = max(1, round(p2_hero.hp * health_multiplier))
                self._logger.info(
                    "[MODE] Hero HP multiplier %.2f: p1=%d->%d p2=%d->%d",
                    health_multiplier,
                    p1_hero.hp,
                    p1_hp,
                    p2_hero.hp,
                    p2_hp,
                )
                p1_hero.hp = p1_hp
                p1_hero.max_hp = p1_hero.hp
                p2_hero.hp = p2_hp
                p2_hero.max_hp = p2_hero.hp

            self._logger.info(
                "[GAME_START] Player %s starting hand (warriors only): %s",
                game_state.p1.user_id,
                [c.name for c in game_state.p1.hand]
            )
            self._logger.info(
                "[GAME_START] Player %s starting hand (warriors only): %s",
                game_state.p2.user_id,
                [c.name for c in game_state.p2.hand]
            )

            # Инициализируем ArenaEnvironment
            self._arena = ArenaEnvironment(
                game_state,
                classic_params=self.mode_config.classic,
            )
            self.current_player_id = game_state.current_turn_owner_id
            self.turn = 1
            self.turn_start_time = time.time()
            self.turn_start_monotonic = time.monotonic()
            self.match_start_time = self.turn_start_time
            self.match_start_monotonic = self.turn_start_monotonic

            p1_actor_type = "bot" if p1_data.get("is_bot", False) else "human"
            p2_actor_type = "bot" if p2_data.get("is_bot", False) else "human"

            def _model_provenance(data: Dict[str, Any]) -> Any:
                supplied = data.get("model_provenance")
                if isinstance(supplied, dict):
                    return supplied
                if not data.get("is_bot"):
                    return None
                getter = getattr(
                    self.berserk_brain,
                    "get_profile_provenance",
                    None,
                )
                if not callable(getter):
                    return None
                try:
                    return getter(
                        str(data.get("difficulty") or self.bot_difficulty),
                        model_id=(
                            str(data.get("brain_profile"))
                            if data.get("brain_profile")
                            else None
                        ),
                    )
                except Exception:
                    self._logger.warning(
                        "Unable to resolve bot model provenance: match=%s",
                        self.match_id,
                        exc_info=True,
                    )
                    return None

            def _aux_provenance(data: Dict[str, Any]) -> Any:
                supplied = data.get("aux_model_provenance")
                if isinstance(supplied, dict):
                    return supplied
                if not data.get("is_bot"):
                    return None
                getter = getattr(
                    self.extra_lr_aux_runtime,
                    "dataset_provenance",
                    None,
                )
                if not callable(getter):
                    return None
                try:
                    return getter(
                        include_policy_assists=(
                            str(data.get("brain_profile") or "")
                            == "extra-lr-v5-ultra"
                        ),
                        include_metronome=True,
                    )
                except Exception:
                    self._logger.warning(
                        "Unable to resolve bot auxiliary provenance: match=%s",
                        self.match_id,
                        exc_info=True,
                    )
                    return None

            self._start_v5_dataset(
                metadata={
                    "p1_deck": self._selection_deck_manifest(
                        p1_data.get("deck_ids") or [],
                        p1_levels,
                        p1_slot_levels,
                    ),
                    "p2_deck": self._selection_deck_manifest(
                        p2_data.get("deck_ids") or [],
                        p2_levels,
                        p2_slot_levels,
                    ),
                    "p1_actor_type": p1_actor_type,
                    "p2_actor_type": p2_actor_type,
                    "battle_tag": f"{p1_actor_type}-vs-{p2_actor_type}",
                    "model_provenance": {
                        "p1": _model_provenance(p1_data),
                        "p2": _model_provenance(p2_data),
                    },
                    "catalog_hash": self._nemesis_catalog_hash(),
                    "aux_model_provenance": {
                        "p1": _aux_provenance(p1_data),
                        "p2": _aux_provenance(p2_data),
                    },
                    "timestamp_features": {
                        "p1_deck_size": len(p1_data.get("deck_ids") or []),
                        "p2_deck_size": len(p2_data.get("deck_ids") or []),
                        "starting_player": (
                            "p1"
                            if game_state.current_turn_owner_id == game_state.p1.user_id
                            else "p2"
                        ),
                    },
                }
            )
            # Replace requested collection levels with the exact effective
            # levels after simplified-card clamping/scaling.
            self.set_v5_dataset_metadata(
                {
                    "p1_deck": self._effective_deck_manifest(p1_deck),
                    "p2_deck": self._effective_deck_manifest(p2_deck),
                }
            )
            try:
                v5_meta = self.get_v5_dataset_snapshot()["meta"]
                cutoff_candidates = [
                    item.get("captured_at")
                    for item in (p1_nemesis_snapshot, p2_nemesis_snapshot)
                    if isinstance(item, dict) and item.get("captured_at")
                ]
                feature_cutoff = max(cutoff_candidates) if cutoff_candidates else v5_meta["started_at"]
                self.nemesis_collector = NemesisBattleCollector.from_v5_meta(
                    v5_meta,
                    feature_cutoff_at=feature_cutoff,
                    extended_by_seat={
                        "p1": p1_nemesis_snapshot,
                        "p2": p2_nemesis_snapshot,
                    },
                )
                self.set_v5_dataset_metadata(
                    {"nemesis_record": self.nemesis_collector.snapshot()}
                )
            except Exception:
                self._logger.warning(
                    "Nemesis collector init failed: match=%s", match_id, exc_info=True
                )
                self.set_v5_dataset_metadata(
                    {
                        "nemesis_collection": {
                            "status": "unavailable",
                            "reason": "collector_init_failed",
                        }
                    }
                )
            
            # Регистрируем в глобальном словаре
            self._active_matches[match_id] = self
            
            self._logger.info("create_match: successfully initialized match_id=%s", match_id)
            
            return {
                "success": True,
                "match_id": match_id,
                "player_ids": self.player_ids,
                "current_player_id": self.current_player_id,
                "starting_player_id": game_state.current_turn_owner_id,
            }
            
        except Exception as exc:
            self._logger.error("create_match failed: %s", exc, exc_info=True)
            return {"success": False, "error": "create_match_failed"}
    
    async def _load_cards_data(self, card_ids: List[Any]) -> Dict[int, Dict[str, Any]]:
        """Загрузить данные карт из БД."""
        if not self._db:
            return {}
        
        cards_data = {}
        for card_id in set(card_ids):
            try:
                cid = int(str(card_id).split(":")[0]) if ":" in str(card_id) else int(card_id)
            except (ValueError, TypeError):
                continue
            
            if cid in self._card_cache:
                cached = self._card_cache[cid]
                if hasattr(cached, "to_dict"):
                    cached = cached.to_dict()
                    self._card_cache[cid] = cached
                cards_data[cid] = cached
                continue
            
            try:
                card_info = await self._db.get_card_info(cid, level=1)
                if card_info:
                    cards_data[cid] = card_info
                    self._card_cache[cid] = card_info
            except Exception as exc:
                self._logger.warning("Failed to load card %s: %s", cid, exc)
        
        return cards_data
    
    async def _load_user_card_levels(self, user_id: int, deck_ids: List[Any]) -> Dict[int, int]:
        """Загрузить уровни карт пользователя."""
        if not self._db:
            return {}
        
        try:
            user_cards = await self._db.get_user_cards(user_id)
            levels = {}
            for card in user_cards:
                card_id = card.get("id")
                level = card.get("level", 1)
                if card_id:
                    levels[int(card_id)] = level
            return levels
        except Exception as exc:
            self._logger.warning("Failed to load user card levels: %s", exc)
            return {}

    def _apply_card_level_mode(self, levels: Dict[int, int], deck_ids: List[Any]) -> Dict[int, int]:
        """Apply the resolved mode's card-level policy to a deck."""
        card_level_mode = self.mode_config.classic.card_level_mode
        if card_level_mode == "normal":
            return levels

        forced_level = 1 if card_level_mode == "disabled" else 10
        adjusted = dict(levels)
        for raw_card_id in deck_ids:
            try:
                card_id = int(str(raw_card_id).split(":")[0])
            except (TypeError, ValueError):
                continue
            adjusted[card_id] = forced_level
        return adjusted
    
    def _extract_hero(self, deck: List[CardInstance]) -> CardInstance:
        """Извлечь героя из колоды или создать дефолтного."""
        from uuid import uuid4
        
        for i, card in enumerate(deck):
            if card.card_type == CardType.HERO:
                self._logger.info(
                    "[ADAPTER] match=%s: найден герой card_id=%s",
                    self.match_id,
                    card.card_id,
                )
                return deck.pop(i)
        
        self._logger.warning(
            "[ADAPTER] match=%s: герой не найден в колоде, создан дефолтный",
            self.match_id,
        )
        return CardInstance(
            instance_id=uuid4(),
            card_id=0,
            name="Герой",
            card_type=CardType.HERO,
            rarity="common",
            mana_cost=0,
            attack=0,
            hp=30,
            max_hp=30,
            mechanics=[],
            is_ready=True,
            description="Дефолтный герой",
        )
    
    # =========================================================================
    # ВЫПОЛНЕНИЕ ДЕЙСТВИЙ (ГЛАВНЫЙ МЕТОД ИНТЕГРАЦИИ)
    # =========================================================================

    @staticmethod
    def _sound_card_snapshot(card: Optional[CardInstance]) -> Optional[dict[str, Any]]:
        """Capture stable card identity before an action mutates hand/board zones."""
        if card is None:
            return None
        card_id_raw = getattr(card, "card_id", None)
        try:
            card_id = int(card_id_raw) if card_id_raw is not None else None
        except (TypeError, ValueError):
            card_id = None
        instance_id_raw = getattr(card, "instance_id", None)
        return {
            "card_id": card_id,
            "instance_id": str(instance_id_raw) if instance_id_raw is not None else None,
            "card_name": str(getattr(card, "name", "") or ""),
            "mechanics": [str(value) for value in (getattr(card, "mechanics", None) or []) if value],
        }

    def _sound_card_snapshot_for_action(
        self,
        user_id: int,
        action: BaseAction,
    ) -> Optional[dict[str, Any]]:
        if not self._arena:
            return None
        player, _opponent = self._arena._resolve_player_pair(user_id)
        if player is None:
            return None
        if isinstance(action, PlayCardAction):
            if 0 <= int(action.hand_index) < len(player.hand):
                return self._sound_card_snapshot(player.hand[int(action.hand_index)])
        if isinstance(action, AttackAction):
            attacker = self._arena._find_unit_by_id(player.board, action.attacker_id)
            return self._sound_card_snapshot(attacker)
        return None

    @staticmethod
    def _is_play_sound_mechanic(mechanic: str) -> bool:
        value = str(mechanic or "")
        if not value or value.startswith("deathrattle"):
            return False
        active_exact = {
            "aoe_freeze",
            "desk_freeze",
            "choose_shield_damage",
            "cast_random_spell",
            "consume_ally",
            "delete_target",
        }
        if value in active_exact:
            return True
        active_prefixes = (
            "battlecry_",
            "spell_",
            "aoe_damage_",
            "damage_",
            "heal_",
            "damage_all",
            "heal_all",
            "buff_all_",
            "summon_",
            "mana_gain_",
            "mana_drain_",
        )
        return value.startswith(active_prefixes)

    @staticmethod
    def _sound_event_id_part(value: Any) -> str:
        text = str(value or "unknown")
        return text.replace(":", "_")

    def _make_sound_event(
        self,
        *,
        turn_number: int,
        action_kind: str,
        card_snapshot: dict[str, Any],
        event: str,
        mechanic: Optional[str] = None,
    ) -> dict[str, Any]:
        instance_id = card_snapshot.get("instance_id")
        event_id_parts = [
            self._sound_event_id_part(self.match_id),
            self._sound_event_id_part(turn_number),
            self._sound_event_id_part(action_kind),
            self._sound_event_id_part(instance_id or card_snapshot.get("card_id")),
            self._sound_event_id_part(event),
        ]
        if mechanic:
            event_id_parts.append(self._sound_event_id_part(mechanic))
        return {
            "event_id": ":".join(event_id_parts),
            "card_id": card_snapshot.get("card_id"),
            "instance_id": instance_id,
            "card_name": str(card_snapshot.get("card_name") or ""),
            "event": event,
            "mechanic": mechanic,
            "side": "player",
            "source": "action",
        }

    @staticmethod
    def _consume_card_feedback_events(state: GameState) -> list[dict[str, Any]]:
        events = list(getattr(state, "pending_card_feedback_events", []) or [])
        if hasattr(state, "pending_card_feedback_events"):
            state.pending_card_feedback_events.clear()
        return events

    @staticmethod
    def _find_card_feedback_event(
        card_snapshot: dict[str, Any],
        mechanic: Optional[str],
        feedback_events: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        instance_id = card_snapshot.get("instance_id")
        card_id = card_snapshot.get("card_id")
        for event in feedback_events:
            if mechanic and str(event.get("mechanic") or "") != str(mechanic):
                continue
            if instance_id and str(event.get("instance_id") or "") == str(instance_id):
                return event
            if card_id is not None and event.get("card_id") == card_id:
                return event
        return None

    def _sound_events_for_action(
        self,
        *,
        action: BaseAction,
        card_snapshot: Optional[dict[str, Any]],
        turn_number: int,
        card_feedback_events: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        if not card_snapshot:
            return []
        feedback_events = card_feedback_events or []
        if isinstance(action, PlayCardAction):
            events = [
                self._make_sound_event(
                    turn_number=turn_number,
                    action_kind="play_card",
                    card_snapshot=card_snapshot,
                    event="deploy",
                )
            ]
            for mechanic in card_snapshot.get("mechanics", []):
                if self._is_play_sound_mechanic(mechanic):
                    event = self._make_sound_event(
                        turn_number=turn_number,
                        action_kind="play_card",
                        card_snapshot=card_snapshot,
                        event="mechanic",
                        mechanic=mechanic,
                    )
                    feedback = self._find_card_feedback_event(card_snapshot, mechanic, feedback_events)
                    if feedback and feedback.get("effect_code"):
                        event["effect_code"] = feedback["effect_code"]
                    events.append(event)
            return events
        if isinstance(action, AttackAction):
            return [
                self._make_sound_event(
                    turn_number=turn_number,
                    action_kind="attack",
                    card_snapshot=card_snapshot,
                    event="attack",
                )
            ]
        return []

    def _preflight_action_error(self, user_id: int, action: BaseAction) -> Optional[str]:
        """Return read-only early errors that happen before core state mutations."""
        if not self._arena:
            return "arena_not_initialized"

        state = self._arena.state
        if state.status != GameStatus.ONGOING:
            return "game_over"

        player, opponent = self._arena._resolve_player_pair(user_id)
        if player is None or opponent is None:
            return "unknown_player"

        if state.current_turn_owner_id != user_id:
            return "not_your_turn"

        known_action = isinstance(action, (PlayCardAction, AttackAction, EndTurnAction))
        if not known_action:
            return None

        try:
            action.validate(state)
        except ValueError as exc:
            return str(exc)
        except Exception:  # noqa: BLE001 - malformed direct actions should stay non-fatal.
            return "invalid_action"

        if isinstance(action, AttackAction):
            attacker = self._arena._find_unit_by_id(player.board, action.attacker_id)
            if not attacker:
                return "attacker_not_found"

            if not attacker.is_ready:
                return "unit_not_ready"

            effective_attack = self._arena._apply_aura_bonuses(attacker, player)
            if effective_attack <= 0:
                return "no_attack"

            if "bypass_taunt" not in attacker.mechanics and has_taunt(opponent.board):
                if action.target_is_hero:
                    return "must_attack_taunt"

                taunt_units = get_taunt_targets(opponent.board)
                if not any(str(unit.instance_id) == action.target_id for unit in taunt_units):
                    return "must_attack_taunt"

            if not action.target_is_hero and not self._arena._find_unit_by_id(opponent.board, action.target_id):
                return "target_not_found"

        return None

    def _begin_v5_action(
        self,
        user_id: int,
        action: BaseAction,
    ) -> int:
        try:
            self._start_v5_dataset()
            source = self._classify_v5_action_source(user_id, action)
            context = self.v5_dataset_recorder.consume_action_context(
                int(user_id),
                fallback_action_json=action.to_dict(),
                decision_source=source,
                actor_type=canonical_v5_actor_type(source),
            )
            return self.v5_dataset_recorder.before_action(
                self,
                user_id=int(user_id),
                action=action,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001 - recording is fail-open.
            self._logger.warning("V5 dataset before_action failed: %s", exc, exc_info=True)
            return -1

    def _arm_current_active_human_clock(self) -> None:
        if not self._arena or self._arena.state.status != GameStatus.ONGOING:
            return
        user_id = int(self._arena.state.current_turn_owner_id)
        if self.is_bot(user_id):
            return
        if self.get_player_replacement_status(user_id) != ReplacementStatus.ACTIVE:
            return
        self.arm_human_decision_clock(user_id)

    def _finish_v5_action(
        self,
        token: int,
        *,
        accepted: bool,
        error: Optional[str] = None,
    ) -> None:
        try:
            outcome = self.v5_dataset_recorder.after_action(
                self,
                token=token,
                accepted=bool(accepted),
                error=error,
            )
            if outcome:
                legacy_index = outcome.get("legacy_action_index")
                if (
                    isinstance(legacy_index, int)
                    and 0 <= legacy_index < len(self._analytics_actions)
                ):
                    legacy_row = self._analytics_actions[legacy_index]
                    legacy_context = legacy_row.setdefault("context_json", {})
                    legacy_context["accepted"] = bool(accepted)
                    legacy_context["error"] = (
                        str(error) if error else None if accepted else "rejected_without_error"
                    )
            if accepted:
                self._arm_current_active_human_clock()
        except Exception as exc:  # noqa: BLE001 - recording is fail-open.
            self._logger.warning("V5 dataset after_action failed: %s", exc, exc_info=True)

    def _record_v5_rejected_action(
        self,
        user_id: int,
        action_json: dict[str, Any],
        error: str,
    ) -> None:
        if not self._arena:
            return
        try:
            self._start_v5_dataset()
            source = self._classify_v5_action_source(int(user_id))
            context = self.v5_dataset_recorder.consume_action_context(
                int(user_id),
                fallback_action_json=action_json,
                decision_source=source,
                actor_type=canonical_v5_actor_type(source),
            )
            token = self.v5_dataset_recorder.record_rejected_action(
                self,
                user_id=int(user_id),
                action_json=action_json,
                context=context,
                error=str(error),
            )
            # ``record_rejected_action`` finishes internally; mirror its
            # accepted/error status into the legacy admin context as well.
            legacy_index = context.get("legacy_action_index")
            if (
                token >= 0
                and isinstance(legacy_index, int)
                and 0 <= legacy_index < len(self._analytics_actions)
            ):
                legacy_context = self._analytics_actions[legacy_index].setdefault(
                    "context_json", {}
                )
                legacy_context["accepted"] = False
                legacy_context["error"] = str(error)
        except Exception as exc:  # noqa: BLE001 - recording is fail-open.
            self._logger.warning("V5 dataset rejected action failed: %s", exc, exc_info=True)
    
    def execute_action(self, user_id: int, action: BaseAction) -> Dict[str, Any]:
        """
        Выполнить действие игрока через core/engine.
        
        Args:
            user_id: ID игрока
            action: Объект действия (PlayCardAction, AttackAction, EndTurnAction)
            
        Returns:
            Результат выполнения с обновленным состоянием
        """
        if not self._arena:
            return {"success": False, "error": "arena_not_initialized"}

        v5_token = self._begin_v5_action(user_id, action)
        preflight_error = self._preflight_action_error(user_id, action)
        if preflight_error is not None:
            self._finish_v5_action(
                v5_token,
                accepted=False,
                error=preflight_error,
            )
            return {"success": False, "error": preflight_error}
        
        import copy

        state_snapshot = copy.deepcopy(self._arena.state)
        rng_snapshot = None
        try:
            rng_snapshot = self._arena._rng.getstate()
        except (AttributeError, TypeError):
            pass
        previous_turn_number = self._arena.state.turn_number
        previous_turn_owner_id = self._arena.state.current_turn_owner_id
        previous_turn_start_time = self.turn_start_time
        previous_turn_start_monotonic = self.turn_start_monotonic
        sound_card_snapshot = self._sound_card_snapshot_for_action(user_id, action)

        # Выполняем действие
        try:
            success, error = self._arena.step(user_id, action)
        except Exception as exc:  # noqa: BLE001
            self._arena.reset_to_state(state_snapshot)
            if rng_snapshot is not None:
                self._arena._rng.setstate(rng_snapshot)
            self.current_player_id = previous_turn_owner_id
            self.turn = previous_turn_number
            self.turn_start_time = previous_turn_start_time
            self.turn_start_monotonic = previous_turn_start_monotonic
            self._logger.error("execute_action failed: %s", exc, exc_info=True)
            self._finish_v5_action(
                v5_token,
                accepted=False,
                error="action_failed",
            )
            return {"success": False, "error": "action_failed"}
        
        if not success:
            self._arena.reset_to_state(state_snapshot)
            if rng_snapshot is not None:
                self._arena._rng.setstate(rng_snapshot)
            self.current_player_id = previous_turn_owner_id
            self.turn = previous_turn_number
            self.turn_start_time = previous_turn_start_time
            self.turn_start_monotonic = previous_turn_start_monotonic
            self._finish_v5_action(
                v5_token,
                accepted=False,
                error=str(error or "action_rejected"),
            )
            return {"success": False, "error": error}
        
        # Обновляем метаданные
        state = self._arena.state
        self.current_player_id = state.current_turn_owner_id
        self.turn = state.turn_number
        card_feedback_events = self._consume_card_feedback_events(state)
        
        # Если был EndTurn, сбрасываем таймер
        if isinstance(action, EndTurnAction):
            self._record_completed_turn(
                previous_turn_number,
                previous_turn_owner_id,
                previous_turn_start_monotonic,
            )
            self.turn_start_time = time.time()
            self.turn_start_monotonic = time.monotonic()

        self._finish_v5_action(v5_token, accepted=True)
        
        # Проверяем завершение игры
        result: Dict[str, Any] = {"success": True}
        sound_events = self._sound_events_for_action(
            action=action,
            card_snapshot=sound_card_snapshot,
            turn_number=previous_turn_number,
            card_feedback_events=card_feedback_events,
        )
        if sound_events:
            result["sound_events"] = sound_events
        
        if state.status != GameStatus.ONGOING:
            self.is_ended = True
            winner_id = None
            if state.status == GameStatus.P1_WIN:
                winner_id = state.p1.user_id
            elif state.status == GameStatus.P2_WIN:
                winner_id = state.p2.user_id
            
            result["game_over"] = True
            result["winner"] = winner_id
            result["winner_id"] = winner_id
            self.finalize_v5_dataset(
                winner_user_id=winner_id,
                status=state.status.value,
            )
        
        # Рассылаем событие обновления (если есть эмиттер)
        if self._event_emitter:
            event_data = {
                "event_type": "state_changed",
                "match_id": self.match_id,
                "action": action.to_dict(),
                "actor_user_id": user_id,
            }
            if sound_events:
                event_data["sound_events"] = sound_events
            self._event_emitter.emit("state_changed", self.match_id, event_data)
        
        return result
    
    # =========================================================================
    # МЕТОДЫ СОВМЕСТИМОСТИ (обертки вокруг execute_action)
    # =========================================================================
    
    def play_card(
        self,
        user_id: int,
        card_id_from_hand: Any,
        target_position: int,
        target_id: Optional[Any] = None,
        target_is_hero: bool = False
    ) -> Dict[str, Any]:
        """
        Розыгрыш карты из руки.
        card_id_from_hand может быть индексом в руке или instance_id карты.
        """
        if not self._arena:
            return {"action": "play_card", "error": "arena_not_initialized"}
        
        state = self._arena.state
        player = state.p1 if state.p1.user_id == user_id else state.p2
        
        # Определяем индекс карты в руке
        hand_index = self._resolve_hand_index(player.hand, card_id_from_hand)
        if hand_index < 0:
            self._record_v5_rejected_action(
                int(user_id),
                {
                    "type": "play_card",
                    "card_ref": card_id_from_hand,
                    "position": target_position,
                    "target_id": target_id,
                    "target_is_hero": bool(target_is_hero),
                },
                "card_not_found_in_hand",
            )
            return {
                "success": False,
                "action": "play_card",
                "error": "card_not_found_in_hand",
            }
        
        # Определяем target_id для core/actions. Если фронтенд уже передал
        # instance_id героя, сохраняем его: герой может быть и союзной целью.
        resolved_target_id = None
        if target_id is not None:
            resolved_target_id = str(target_id)
        elif target_is_hero:
            opponent = state.p2 if state.p1.user_id == user_id else state.p1
            resolved_target_id = str(opponent.hero.instance_id)
        
        action = PlayCardAction(
            hand_index=hand_index,
            target_id=resolved_target_id,
            position=target_position,
        )
        
        result = self.execute_action(user_id, action)
        result["action"] = "play_card"
        return result
    
    def attack_target(
        self,
        user_id: int,
        attacker_id: Any,
        target_id: Any,
        target_is_hero: bool = False,
    ) -> Dict[str, Any]:
        """Атака существом."""
        if not self._arena:
            return {"action": "attack", "error": "arena_not_initialized"}
        
        action = AttackAction(
            attacker_id=str(attacker_id),
            target_id=str(target_id) if not target_is_hero else None,
            target_is_hero=target_is_hero,
        )
        
        result = self.execute_action(user_id, action)
        result["action"] = "attack"
        return result
    
    def end_turn(self, user_id: int) -> Dict[str, Any]:
        """Завершение хода."""
        if not self._arena:
            return {"action": "end_turn", "error": "arena_not_initialized"}

        action = EndTurnAction()
        result = self.execute_action(user_id, action)
        result["action"] = "end_turn"
        return result

    def mana_draw(self, user_id: int) -> Dict[str, Any]:
        """Добор одной карты за ману (player-initiated). См. core/engine.py
        ArenaEnvironment._handle_mana_draw и docs/CYCLE_DRAW.md."""
        if not self._arena:
            return {"action": "mana_draw", "error": "arena_not_initialized"}

        action = ManaDrawAction()
        result = self.execute_action(user_id, action)
        result["action"] = "mana_draw"
        return result

    def _talkie_extra_pass_for_user(self, user_id: int) -> str:
        if int(user_id) == int(self._p1_id):
            mode = self._p1_extra_pass
        elif int(user_id) == int(self._p2_id):
            mode = self._p2_extra_pass
        else:
            mode = None
        normalized = str(mode or "inactive").lower()
        return normalized if normalized in TALKIE_TIER_LIMITS else "inactive"

    def _talkie_limits_for_user(self, user_id: int) -> dict[str, float | int | None]:
        return TALKIE_TIER_LIMITS[self._talkie_extra_pass_for_user(user_id)]

    def set_talkie_enabled(self, user_id: int, enabled: bool) -> dict[str, Any]:
        user_id_int = int(user_id)
        self._talkie_enabled_by_user[user_id_int] = bool(enabled)
        return {
            "success": True,
            "user_id": user_id_int,
            "enabled": self.talkie_enabled_for(user_id_int),
        }

    def talkie_enabled_for(self, user_id: int) -> bool:
        return self._talkie_enabled_by_user.get(int(user_id), True)

    def should_deliver_talkie_to(self, user_id: int) -> bool:
        return self.talkie_enabled_for(user_id)

    def register_talkie(
        self,
        user_id: int,
        talkie_id: Any,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        user_id_int = int(user_id)
        talkie_id_str = str(talkie_id)

        catalog_entry = TALKIE_CATALOG.get(talkie_id_str)
        if catalog_entry is None:
            return {"success": False, "error": "invalid_talkie"}

        if not self.talkie_enabled_for(user_id_int):
            return {"success": False, "error": "talkie_disabled"}

        if int(self.current_player_id or 0) != user_id_int:
            return {"success": False, "error": "not_your_turn"}

        current_time = float(time.monotonic() if now is None else now)
        limits = self._talkie_limits_for_user(user_id_int)
        max_per_turn = int(limits["max_per_turn"] or 0)
        cooldown_seconds = limits["cooldown_seconds"]
        turn_number = int(self.turn)
        usage_key = (user_id_int, turn_number)
        usage = self._talkie_usage_by_turn.setdefault(usage_key, [])
        remaining = max(0, max_per_turn - len(usage))

        if len(usage) >= max_per_turn:
            return {
                "success": False,
                "error": "talkie_limit_reached",
                "remaining": 0,
            }

        if cooldown_seconds is not None and usage:
            elapsed = current_time - usage[-1]
            retry_after = float(cooldown_seconds) - elapsed
            if retry_after > 0:
                return {
                    "success": False,
                    "error": "talkie_cooldown",
                    "retry_after": round(retry_after, 3),
                    "remaining": remaining,
                }

        usage.append(current_time)
        self._talkie_event_seq += 1
        remaining = max(0, max_per_turn - len(usage))
        return {
            "success": True,
            "remaining": remaining,
            "event": {
                "event_id": f"{self.match_id}:{turn_number}:{user_id_int}:{self._talkie_event_seq}",
                "match_id": self.match_id,
                "sender_id": user_id_int,
                "talkie_id": talkie_id_str,
                "sound": catalog_entry["sound"],
                "turn": turn_number,
                "remaining": remaining,
            },
        }
    
    def _resolve_hand_index(self, hand: List[CardInstance], card_ref: Any) -> int:
        """
        Определить индекс карты в руке.
        card_ref может быть:
        - int индексом напрямую
        - instance_id (UUID или строка)
        - card_id (int)
        """
        # Прямой индекс
        if isinstance(card_ref, int) and 0 <= card_ref < len(hand):
            return card_ref
        
        # Поиск по instance_id или card_id
        card_ref_str = str(card_ref)
        for i, card in enumerate(hand):
            if str(card.instance_id) == card_ref_str:
                return i
            if str(card.card_id) == card_ref_str:
                return i
        
        return -1
    
    # =========================================================================
    # ПОЛУЧЕНИЕ СОСТОЯНИЯ ДЛЯ ФРОНТЕНДА
    # =========================================================================
    
    def get_full_state(self, viewer_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Сериализация состояния для отправки на фронтенд.
        Включает legal_actions для текущего игрока.
        """
        if not self._arena:
            return {
                "match_id": self.match_id,
                "error": "arena_not_initialized",
            }
        
        state = self._arena.state
        p1 = state.p1
        p2 = state.p2
        
        viewer_is_p1 = viewer_id is not None and p1.user_id == viewer_id
        viewer_is_p2 = viewer_id is not None and p2.user_id == viewer_id
        known_viewer = viewer_is_p1 or viewer_is_p2

        # Определяем, кто player, кто opponent относительно viewer.
        # Без распознанного viewer_id сохраняем стабильную p1-perspective, но скрываем обе руки.
        if viewer_is_p2:
            player_state, opponent_state = p2, p1
            player_name, opponent_name = self._p2_name, self._p1_name
            player_avatar, opponent_avatar = self._p2_avatar_url, self._p1_avatar_url
            player_title, opponent_title = self._p2_title, self._p1_title
            player_rarity, opponent_rarity = self._p2_rarity, self._p1_rarity
            player_extra_pass, opponent_extra_pass = self._p2_extra_pass, self._p1_extra_pass
            player_glow_disabled, opponent_glow_disabled = self._p2_nickname_glow_disabled, self._p1_nickname_glow_disabled
            player_hide_id, opponent_hide_id = self._p2_hide_player_id_public, self._p1_hide_player_id_public
            player_background, opponent_background = self._p2_background_url, self._p1_background_url
        else:
            player_state, opponent_state = p1, p2
            player_name, opponent_name = self._p1_name, self._p2_name
            player_avatar, opponent_avatar = self._p1_avatar_url, self._p2_avatar_url
            player_title, opponent_title = self._p1_title, self._p2_title
            player_rarity, opponent_rarity = self._p1_rarity, self._p2_rarity
            player_extra_pass, opponent_extra_pass = self._p1_extra_pass, self._p2_extra_pass
            player_glow_disabled, opponent_glow_disabled = self._p1_nickname_glow_disabled, self._p2_nickname_glow_disabled
            player_hide_id, opponent_hide_id = self._p1_hide_player_id_public, self._p2_hide_player_id_public
            player_background, opponent_background = self._p1_background_url, self._p2_background_url
        
        waiting_for_players = self.is_waiting_for_players()
        viewer_can_control = (
            known_viewer
            and getattr(player_state, "replacement_status", ReplacementStatus.ACTIVE) == ReplacementStatus.ACTIVE
        )
        is_my_turn = (
            known_viewer
            and state.current_turn_owner_id == viewer_id
            and not waiting_for_players
            and viewer_can_control
        )
        
        # Сериализуем legal_actions для viewer (если это его ход)
        legal_actions_json = []
        if known_viewer and is_my_turn:
            legal_actions = self._arena.get_legal_actions(viewer_id)
            legal_actions_json = [self._serialize_action(a) for a in legal_actions]

        required_ready_user_ids = self.required_ready_user_ids()
        waiting_for_user_ids = [
            user_id for user_id in required_ready_user_ids
            if user_id not in self.client_ready_users
        ]
        
        return {
            "match_id": self.match_id,
            "turn": state.turn_number,
            "current_player_id": state.current_turn_owner_id,
            "is_my_turn": is_my_turn,
            "player_ids": [p1.user_id, p2.user_id],
            "viewer_id": viewer_id,
            "match_status": "waiting_for_players" if waiting_for_players else "active",
            "battle_started": not waiting_for_players,
            "ready_user_ids": sorted(self.client_ready_users),
            "waiting_for_user_ids": waiting_for_user_ids,
            "ready_count": len(self.client_ready_users.intersection(required_ready_user_ids)),
            "required_ready_count": len(required_ready_user_ids),
            "is_ended": self.is_ended,
            "game_over": state.status != GameStatus.ONGOING,
            "winner_id": self._get_winner_id(),
            "turn_time_remaining": self.get_turn_time_remaining(),
            "turn_duration": self.turn_duration,
            "turn_time_history": self._serialize_turn_time_history(viewer_id),
            "game_mode": self.game_mode,
            "ruleset": self.ruleset,
            "mode_config": serialize_mode_config(self.mode_config),
            "sudden_death": self._serialize_sudden_death_state(player_state, opponent_state),
            
            # HP героев (для логов и быстрой проверки)
            "player1_hp": p1.hero.hp,
            "player2_hp": p2.hero.hp,
            
            # Состояние viewer'а (player)
            "player": self._serialize_player_state(player_state, player_name, player_avatar, player_title, player_rarity,
                                                    player_extra_pass, player_background, show_hand=known_viewer,
                                                    nickname_glow_disabled=player_glow_disabled,
                                                    hide_player_id_public=player_hide_id),
            
            # Состояние оппонента (скрываем содержимое руки)
            "opponent": self._serialize_player_state(opponent_state, opponent_name, opponent_avatar, opponent_title, opponent_rarity,
                                                      opponent_extra_pass, opponent_background, show_hand=False,
                                                      nickname_glow_disabled=opponent_glow_disabled,
                                                      hide_player_id_public=opponent_hide_id),
            
            # КРИТИЧНО: Список доступных действий
            "legal_actions": legal_actions_json,
            
            # История последних 100 действий (преобразуем в формат для клиента)
            "action_history": self._serialize_action_history(state.action_history, viewer_id),
        }

    def _serialize_sudden_death_state(
        self,
        player_state: CorePlayerState,
        opponent_state: CorePlayerState,
    ) -> Dict[str, Any]:
        """Serialize viewer-relative Sudden Death counters for UI hints."""
        classic = self.mode_config.classic
        if not classic.sudden_death_enabled or not self._arena:
            return {"enabled": False}

        state = self._arena.state

        def _turn_count(ps: CorePlayerState) -> int:
            return int(state.sudden_death_turns_by_player.get(int(ps.user_id), 0))

        def _next_damage(turn_count: int) -> int:
            return classic.sudden_death_damage_start + turn_count * classic.sudden_death_damage_step

        def _turn_damage(ps: CorePlayerState, turn_count: int) -> int | None:
            if int(ps.user_id) != int(getattr(state, "current_turn_owner_id", 0) or 0):
                return None
            applied_turn_count = max(1, turn_count)
            return (
                classic.sudden_death_damage_start
                + (applied_turn_count - 1) * classic.sudden_death_damage_step
            )

        player_turns = _turn_count(player_state)
        opponent_turns = _turn_count(opponent_state)
        return {
            "enabled": True,
            "damage_start": classic.sudden_death_damage_start,
            "damage_step": classic.sudden_death_damage_step,
            "player_turn_count": player_turns,
            "opponent_turn_count": opponent_turns,
            "player_turn_damage": _turn_damage(player_state, player_turns),
            "opponent_turn_damage": _turn_damage(opponent_state, opponent_turns),
            "player_next_damage": _next_damage(player_turns),
            "opponent_next_damage": _next_damage(opponent_turns),
        }
    
    def _serialize_player_state(
        self, 
        ps: CorePlayerState, 
        name: str, 
        avatar_url: Optional[str],
        title: str = "",
        rarity: str = "",
        extra_pass: Optional[str] = None,
        background_url: Optional[str] = None,
        show_hand: bool = True,
        nickname_glow_disabled: bool = False,
        hide_player_id_public: bool = False,
    ) -> Dict[str, Any]:
        """Сериализация состояния игрока."""
        hand_data = []
        if show_hand:
            hand_data = [self._serialize_card(c) for c in ps.hand]
        else:
            # Для оппонента показываем только количество карт
            hand_data = [{"hidden": True} for _ in ps.hand]
        
        return {
            "user_id": ps.user_id,
            "name": name,
            "avatar_url": avatar_url,
            "title": title,
            "rarity": rarity,
            "extra_pass": extra_pass,
            "nickname_glow_disabled": bool(nickname_glow_disabled),
            "hide_player_id_public": bool(hide_player_id_public),
            "background_url": background_url,
            "is_bot": ps.is_bot,
            "replacement_status": getattr(ps.replacement_status, "value", str(ps.replacement_status)),
            "mana": ps.mana,
            "max_mana": ps.max_mana,
            # Счётчик доборов за ману в текущем ходу — нужен клиенту, чтобы
            # считать следующую стоимость (MANA_DRAW_BASE*(count+1)). Только
            # для своего состояния (show_hand=True), чтобы не утекало инфо об
            # оппоненте.
            "mana_draw_count_this_turn": ps.mana_draw_count_this_turn if show_hand else 0,
            "trophies": ps.trophies,
            "hero": self._serialize_card(ps.hero),
            "hand": hand_data,
            "hand_count": len(ps.hand),
            "deck_count": len(ps.deck),
            "board": [self._serialize_card(c, owner_id=ps.user_id) for c in ps.board],
        }
    
    def _serialize_card(self, card: CardInstance, owner_id: Optional[int] = None) -> Dict[str, Any]:
        """Сериализация карты для JSON."""
        return {
            "instance_id": str(card.instance_id),
            "card_id": card.card_id,
            "name": card.name,
            "description": card.description,
            "card_type": card.card_type.value,
            "rarity": card.rarity,
            "level": card.level,
            "mana": card.mana_cost,
            "mana_cost": card.mana_cost,
            "attack": card.attack,
            "atk": card.attack,
            "hp": card.hp,
            "hp_current": card.hp,
            "max_hp": card.max_hp,
            "mechanics": card.mechanics,
            "is_ready": card.is_ready,
            "can_attack": card.is_ready and card.attack > 0,
            "is_asleep": card.is_asleep,
            "is_frozen": card.is_frozen,
            "mechanics_desc": card.mechanics_desc,
            "owner_id": owner_id,
            "image": card_asset_url(card.card_id) if card.card_id else card_asset_url(9),
        }
    
    def _serialize_action(self, action: BaseAction) -> Dict[str, Any]:
        """Сериализация действия для JSON."""
        data = action.to_dict()
        # Добавляем дополнительные поля для удобства фронтенда
        if isinstance(action, PlayCardAction):
            data["hand_index"] = action.hand_index
        elif isinstance(action, AttackAction):
            data["attacker_id"] = action.attacker_id
            data["target_id"] = action.target_id
            data["target_is_hero"] = action.target_is_hero
        return data
    
    def _serialize_action_history(
        self, 
        action_history: List[tuple[str, str]], 
        viewer_id: Optional[int]
    ) -> List[Dict[str, str]]:
        """
        Сериализация истории действий для клиента.
        Преобразует (type, text) в формат с правильными метками 'Вы'/'Противник'.
        
        Args:
            action_history: Список (type, text) из GameState
            viewer_id: ID клиента, относительно которого маркируются P1/P2-события
            
        Returns:
            Список словарей {"type": "player"/"opponent"/"system", "text": "описание"}
        """
        result = []
        for log_type, text in action_history:
            # В core log_type привязан к P1/P2, а клиенту нужна перспектива viewer.
            if log_type == "player" and viewer_id is not None and self._arena:
                actor_id = self._arena.state.p1.user_id
                final_type = "player" if actor_id == viewer_id else "opponent"
                prefix = "Вы" if final_type == "player" else "Противник"
            elif log_type == "opponent" and viewer_id is not None and self._arena:
                actor_id = self._arena.state.p2.user_id
                final_type = "player" if actor_id == viewer_id else "opponent"
                prefix = "Вы" if final_type == "player" else "Противник"
            elif log_type == "player":
                final_type = "player"
                prefix = "Вы"
            elif log_type == "opponent":
                final_type = "opponent"
                prefix = "Противник"
            else:
                final_type = "system"
                prefix = ""
            
            # Формируем итоговый текст с префиксом
            if prefix:
                # Если текст содержит ":" (формат "Карта: эффект"), добавляем префикс перед двоеточием
                if ":" in text and not text.startswith(prefix):
                    # Разделяем на часть до : и после
                    parts = text.split(":", 1)
                    final_text = f"{prefix} {parts[0]}:{parts[1]}"
                else:
                    # Простой текст - добавляем префикс в начало
                    final_text = f"{prefix} {text}"
            else:
                final_text = text
            
            result.append({
                "type": final_type,
                "text": final_text
            })
        
        return result
    
    def _get_winner_id(self) -> Optional[int]:
        """Определить победителя. Возвращает None при ничьей."""
        if not self._arena:
            return None
        status = self._arena.state.status
        if status == GameStatus.P1_WIN:
            return self._arena.state.p1.user_id
        elif status == GameStatus.P2_WIN:
            return self._arena.state.p2.user_id
        return None
    
    # =========================================================================
    # LEGAL ACTIONS (удобные методы)
    # =========================================================================
    
    def check_game_over(self) -> Dict[str, Any]:
        """Проверить завершение игры. Возвращает {'game_over': bool, 'winner_id': int|None}."""
        if not self._arena:
            return {"game_over": False, "winner_id": None}
        state = self._arena.state
        game_over = state.status != GameStatus.ONGOING
        winner_id = self._get_winner_id() if game_over else None
        if game_over:
            self.is_ended = True
        return {"game_over": game_over, "winner_id": winner_id}

    # =========================================================================
    
    def get_legal_actions(self, user_id: int) -> List[Dict[str, Any]]:
        """Получить список доступных действий для игрока (сериализованный)."""
        if not self._arena:
            return []
        actions = self._arena.get_legal_actions(user_id)
        return [self._serialize_action(a) for a in actions]
    
    def get_preview_delta(self, action: BaseAction) -> Dict[str, int]:
        """
        Получить предпросмотр изменений HP для действия.
        
        Args:
            action: Действие для предпросмотра
            
        Returns:
            Словарь {instance_id: delta_hp}
        """
        if not self._arena:
            return {}
        return self._arena.get_preview_delta(action)
    
    # =========================================================================
    # ТАЙМЕРЫ И СЛУЖЕБНЫЕ МЕТОДЫ
    # =========================================================================
    
    def get_current_player_id(self) -> Optional[int]:
        """Возвращает ID текущего игрока."""
        if self._arena:
            return self._arena.state.current_turn_owner_id
        return self.current_player_id
    
    def get_turn_time_remaining(self) -> float:
        """Возвращает оставшееся время хода."""
        if self.is_waiting_for_players():
            return float(self.turn_duration)
        if self.turn_start_time is None:
            return float(self.turn_duration)
        start = self.turn_start_monotonic if self.turn_start_monotonic is not None else self.turn_start_time
        elapsed = time.monotonic() - start if self.turn_start_monotonic is not None else time.time() - start
        remaining = max(0, self.turn_duration - elapsed)
        return remaining

    def _record_completed_turn(
        self,
        turn_number: int,
        player_id: Optional[int],
        started_at: Optional[float],
        *,
        now: Optional[float] = None,
    ) -> None:
        if player_id is None or started_at is None:
            return
        current_time = time.monotonic() if now is None else now
        elapsed = max(0.0, min(float(self.turn_duration), current_time - started_at))
        self.turn_time_history.append({
            "turn": int(turn_number or 0),
            "player_id": int(player_id),
            "elapsed_seconds": round(elapsed, 1),
        })
        self.turn_time_history = self.turn_time_history[-12:]

    def _serialize_turn_time_history(self, viewer_id: Optional[int]) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for item in self.turn_time_history[-8:]:
            player_id = item.get("player_id")
            side = "player" if viewer_id is not None and player_id == viewer_id else "opponent"
            history.append({
                "turn": item.get("turn"),
                "player_id": player_id,
                "side": side,
                "elapsed_seconds": item.get("elapsed_seconds"),
            })
        return history
    
    def is_turn_expired(self) -> bool:
        """Проверяет истечение времени хода."""
        if self.is_waiting_for_players():
            return False
        return self.get_turn_time_remaining() <= 0
    
    def is_bot(self, player_id: Optional[int]) -> bool:
        """Проверяет, является ли игрок ботом."""
        if not self._arena or player_id is None:
            return False
        state = self._arena.state
        if state.p1.user_id == player_id:
            return state.p1.is_bot
        elif state.p2.user_id == player_id:
            return state.p2.is_bot
        return False
    
    def is_current_player_bot(self) -> bool:
        """Проверяет, является ли текущий игрок ботом."""
        return self.is_bot(self.get_current_player_id())
    
    def mark_timeout(self, player_id: int) -> None:
        """
        Отмечает таймаут игрока. При достижении 2 таймаутов переводит в статус AFK.
        
        Args:
            player_id: ID игрока, у которого произошёл таймаут
        """
        from core.state import ReplacementStatus
        
        if not self._arena:
            return
        
        state = self._arena.state
        
        # Определяем игрока и увеличиваем счётчик
        if state.p1.user_id == player_id:
            self.p1_consecutive_timeouts += 1
            player = state.p1
            timeout_count = self.p1_consecutive_timeouts
        elif state.p2.user_id == player_id:
            self.p2_consecutive_timeouts += 1
            player = state.p2
            timeout_count = self.p2_consecutive_timeouts
        else:
            self._logger.warning("mark_timeout: unknown player_id=%s", player_id)
            return
        
        self._logger.info(
            "[TIMEOUT] Match: %s | Player: %s | Consecutive timeouts: %d",
            self.match_id, player_id, timeout_count
        )
        
        # При 2+ таймаутах переводим в AFK
        if timeout_count >= 2 and player.replacement_status == ReplacementStatus.ACTIVE:
            self.set_player_replacement_status(
                player_id,
                ReplacementStatus.AFK,
                reason="consecutive_timeouts",
            )
            self._logger.warning(
                "[AFK_DETECTED] Match: %s | Player: %s marked as AFK (2+ timeouts)",
                self.match_id, player_id
            )
            self._logger.info("Player %s marked AFK after %s timeouts", player_id, timeout_count)

    def reset_player_timeouts(self, player_id: int) -> None:
        """Reset consecutive natural timer expirations for a player."""
        if not self._arena:
            return

        state = self._arena.state
        if state.p1.user_id == player_id:
            self.p1_consecutive_timeouts = 0
        elif state.p2.user_id == player_id:
            self.p2_consecutive_timeouts = 0
        else:
            self._logger.warning("reset_player_timeouts: unknown player_id=%s", player_id)

    def get_player_replacement_status(self, user_id: int) -> "ReplacementStatus":
        """Return ACTIVE/AFK/SURRENDERED for a participant."""
        from core.state import ReplacementStatus

        if not self._arena:
            return ReplacementStatus.ACTIVE

        state = self._arena.state
        if state.p1.user_id == user_id:
            return getattr(state.p1, "replacement_status", ReplacementStatus.ACTIVE)
        if state.p2.user_id == user_id:
            return getattr(state.p2, "replacement_status", ReplacementStatus.ACTIVE)
        return ReplacementStatus.ACTIVE

    def restore_player_control(self, user_id: int) -> bool:
        """
        Return an AFK-replaced human to ACTIVE control without rewinding battle state.

        SURRENDERED is intentionally final and is not restored.
        """
        from core.state import ReplacementStatus

        if not self._arena:
            return False

        status = self.get_player_replacement_status(user_id)
        if status == ReplacementStatus.SURRENDERED:
            return False
        if status == ReplacementStatus.AFK:
            self.set_player_replacement_status(user_id, ReplacementStatus.ACTIVE)
        self.reset_player_timeouts(user_id)
        return status == ReplacementStatus.AFK

    def mark_player_activity(self, user_id: int) -> None:
        """Any real player action/reconnect proves activity and clears AFK timeout counters."""
        self.restore_player_control(user_id)
    
    def mark_surrender(self, player_id: int) -> None:
        """
        Отмечает сдачу игрока. Игрок переводится в статус SURRENDERED,
        но бой продолжается (за него играет бот).
        
        Args:
            player_id: ID игрока, который сдался
        """
        from core.state import ReplacementStatus
        
        if not self._arena:
            return
        
        state = self._arena.state
        
        # Определяем игрока
        if state.p1.user_id == player_id:
            player = state.p1
        elif state.p2.user_id == player_id:
            player = state.p2
        else:
            self._logger.warning("mark_surrender: unknown player_id=%s", player_id)
            return
        
        # Переводим в статус SURRENDERED и сохраняем отдельное control event.
        self.set_player_replacement_status(
            player_id,
            ReplacementStatus.SURRENDERED,
            reason="surrender",
        )
        
        self._logger.warning(
            "[SURRENDER] Player %s surrendered. Bot will continue playing.",
            player_id
        )
        self._logger.info("Player %s surrendered; replacement bot takes control", player_id)
    
    def _requires_ready_barrier(self) -> bool:
        """Any match with a human starts only after all human clients are ready."""
        if not self._arena:
            return False
        required_ids = self.required_ready_user_ids()
        return len(required_ids) >= 1

    def required_ready_user_ids(self) -> set[int]:
        """Human participants that must report ``client_ready`` before actions unlock."""
        if not self._arena:
            return set()
        state = self._arena.state
        required: set[int] = set()
        for player in (state.p1, state.p2):
            if (
                not getattr(player, "is_bot", False)
                and getattr(player, "replacement_status", ReplacementStatus.ACTIVE) == ReplacementStatus.ACTIVE
            ):
                try:
                    raw_user_id = getattr(player, "user_id", None)
                    if raw_user_id is None:
                        continue
                    required.add(int(raw_user_id))
                except (TypeError, ValueError):
                    continue
        return required

    def is_waiting_for_players(self) -> bool:
        """Return True while required human clients have not received prebattle state."""
        if self.is_ended or not self._requires_ready_barrier():
            return False
        return not self.required_ready_user_ids().issubset(self.client_ready_users)

    def mark_client_ready(self, user_id: Optional[int] = None) -> dict[str, Any]:
        """Помечает клиента готовым и запускает таймер после готовности всех людей."""
        was_ready = bool(self.client_ready)
        if user_id is not None:
            try:
                self.client_ready_users.add(int(user_id))
            except (TypeError, ValueError):
                pass

        required_ids = self.required_ready_user_ids()
        if not self._requires_ready_barrier():
            self.client_ready = True
        elif required_ids.issubset(self.client_ready_users):
            if not self.client_ready:
                self.turn_start_time = time.time()
                self.turn_start_monotonic = time.monotonic()
                self.match_start_time = self.turn_start_time
                self.match_start_monotonic = self.turn_start_monotonic
            self.client_ready = True
        else:
            self.client_ready = False

        if self.client_ready and not was_ready:
            self.mark_v5_battle_started(reason="client_ready")
            self._arm_current_active_human_clock()

        ready_count = len(self.client_ready_users.intersection(required_ids))
        return {
            "all_ready": bool(self.client_ready),
            "ready_user_ids": sorted(self.client_ready_users),
            "waiting_for_user_ids": sorted(required_ids - self.client_ready_users),
            "ready_count": ready_count,
            "required_ready_count": len(required_ids),
        }
    
    def set_player_replacement_status(
        self,
        user_id: int,
        status: "ReplacementStatus",
        *,
        reason: Optional[str] = None,
    ) -> None:
        """
        Устанавливает статус замены для игрока.
        Используется для мгновенного перевода в AFK при разрыве сокета.
        
        Args:
            user_id: ID игрока
            status: Новый статус (ReplacementStatus.AFK, SURRENDERED, ACTIVE)
        """
        from core.state import ReplacementStatus
        
        if not self._arena:
            self._logger.warning("set_player_replacement_status: arena not initialized")
            return
        
        state = self._arena.state
        
        # Определяем игрока и меняем статус
        player = None
        if state.p1.user_id == user_id:
            player = state.p1
            previous_status = state.p1.replacement_status
            state.p1.replacement_status = status
            self._logger.info(
                "[STATUS_CHANGE] Match: %s | Player: %s | New status: %s",
                self.match_id, user_id, status.value
            )
        elif state.p2.user_id == user_id:
            player = state.p2
            previous_status = state.p2.replacement_status
            state.p2.replacement_status = status
            self._logger.info(
                "[STATUS_CHANGE] Match: %s | Player: %s | New status: %s",
                self.match_id, user_id, status.value
            )
        else:
            self._logger.warning(
                "set_player_replacement_status: unknown player_id=%s in match=%s",
                user_id, self.match_id
            )
            return

        if player is not None and previous_status != status:
            try:
                self._start_v5_dataset()
                self.v5_dataset_recorder.record_control_change(
                    self,
                    user_id=int(user_id),
                    previous_status=previous_status,
                    new_status=status,
                    reason=reason,
                )
                if status == ReplacementStatus.ACTIVE:
                    self.censor_human_decision_clock(
                        int(user_id), "control_restored"
                    )
                else:
                    self.censor_human_decision_clock(
                        int(user_id),
                        str(reason or f"control_changed_{status.value}"),
                    )
            except Exception as exc:  # noqa: BLE001 - telemetry is fail-open.
                self._logger.warning(
                    "V5 control-change recording failed: %s",
                    exc,
                    exc_info=True,
                )
    
    # =========================================================================
    # СОВМЕСТИМОСТЬ СО СТАРЫМ API (property-подобные атрибуты)
    # =========================================================================
    
    @property
    def p1_state(self) -> Any:
        """Совместимость: возвращает обертку над p1."""
        if self._arena:
            return _LegacyPlayerStateWrapper(
                self._arena.state.p1, 
                self._p1_name, 
                self._p1_avatar_url
            )
        return _EmptyLegacyState(self._p1_id, self._p1_name)
    
    @property
    def p2_state(self) -> Any:
        """Совместимость: возвращает обертку над p2."""
        if self._arena:
            return _LegacyPlayerStateWrapper(
                self._arena.state.p2,
                self._p2_name,
                self._p2_avatar_url
            )
        return _EmptyLegacyState(self._p2_id, self._p2_name)
    
    def get_player_state(self, user_id: int) -> Any:
        """Возвращает состояние игрока."""
        if self._arena:
            if self._arena.state.p1.user_id == user_id:
                return self.p1_state
            if self._arena.state.p2.user_id == user_id:
                return self.p2_state
            return None
        return _EmptyLegacyState(user_id, "Игрок")
    
    def get_opponent_state(self, user_id: int) -> Any:
        """Возвращает состояние противника."""
        if self._arena:
            if self._arena.state.p1.user_id == user_id:
                return self.p2_state
            return self.p1_state
        return _EmptyLegacyState(0, "Противник")
    
    def execute_bot_action(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет действие бота (из словаря)."""
        if not self._arena:
            return {"error": "arena_not_initialized"}
        
        action_type = action_dict.get("type")
        
        if action_type == "play_card":
            action = PlayCardAction(
                hand_index=action_dict.get("hand_index", 0),
                target_id=action_dict.get("target_id"),
                position=action_dict.get("position"),
            )
        elif action_type == "attack":
            action = AttackAction(
                attacker_id=str(action_dict.get("attacker_id", "")),
                target_id=action_dict.get("target_id"),
                target_is_hero=action_dict.get("target_is_hero", False),
            )
        elif action_type == "end_turn":
            action = EndTurnAction()
        elif action_type == "mana_draw":
            action = ManaDrawAction()
        else:
            current_player_id = self.get_current_player_id()
            if current_player_id is not None:
                self._record_v5_rejected_action(
                    int(current_player_id),
                    dict(action_dict),
                    f"unknown_action_type: {action_type}",
                )
            return {
                "success": False,
                "error": f"unknown_action_type: {action_type}",
            }
        
        return self.execute_action(self.get_current_player_id(), action)

    # ── Analytics helpers ──

    @staticmethod
    def _card_params_payload(card: CardInstance) -> dict[str, Any]:
        return {
            "schema": TRAIN_V3_CARD_PARAMS_SCHEMA,
            "type": getattr(card.card_type, "value", str(card.card_type)),
            "mana_cost": int(getattr(card, "mana_cost", 0) or 0),
            "attack": int(getattr(card, "attack", 0) or 0),
            "hp": int(getattr(card, "hp", 0) or 0),
            "max_hp": int(getattr(card, "max_hp", 0) or 0),
            "mechanics": list(getattr(card, "mechanics", None) or []),
            "is_ready": bool(getattr(card, "is_ready", False)),
            "is_frozen": bool(getattr(card, "is_frozen", False)),
            "level": int(getattr(card, "level", 1) or 1),
        }

    @classmethod
    def _card_params_slot_payload(
        cls,
        card: CardInstance,
        *,
        slot: int,
        zone: str,
        hand_index: Optional[int] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "slot": int(slot),
            "zone": zone,
            "card": cls._card_params_payload(card),
        }
        if hand_index is not None:
            payload["hand_index"] = int(hand_index)
        return payload

    @classmethod
    def _deck_param_snapshot(cls, cards: list[CardInstance]) -> dict[str, Any]:
        return {
            "schema": TRAIN_V3_DECK_PARAMS_SCHEMA,
            "card_params_schema": TRAIN_V3_CARD_PARAMS_SCHEMA,
            "cards": [
                cls._card_params_slot_payload(card, slot=idx, zone="initial_deck")
                for idx, card in enumerate(cards)
            ],
        }

    @staticmethod
    def _snapshot_card(card: CardInstance) -> dict:
        card_params = BattleEngine._card_params_payload(card)
        return {
            "instance_id": str(card.instance_id),
            "id": card.card_id,
            "card_id": card.card_id,
            "name": card.name,
            "type": getattr(card.card_type, "value", str(card.card_type)),
            "card_type": getattr(card.card_type, "value", str(card.card_type)),
            "rarity": card.rarity,
            "level": card.level,
            "mana": getattr(card, "mana_cost", 0),
            "mana_cost": card_params["mana_cost"],
            "atk": card.attack,
            "attack": card_params["attack"],
            "hp": card.hp,
            "max_hp": card.max_hp,
            "is_ready": card.is_ready,
            "is_frozen": card.is_frozen,
            "mechanics": list(card.mechanics or []),
            "card_params": card_params,
        }

    def build_analytics_snapshot(self) -> Optional[dict]:
        if not self._arena:
            return None
        st = self._arena.state
        def _player(p):
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
        return {
            "turn": st.turn_number,
            "current_player": st.current_turn_owner_id,
            "p1": _player(st.p1),
            "p2": _player(st.p2),
        }

    def _build_analytics_action_context(
        self,
        user_id: int,
        acting_player: int,
        action_json: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._arena:
            return {}
        st = self._arena.state
        player = st.p1 if acting_player == 1 else st.p2
        acting_hand = [
            self._card_params_slot_payload(card, slot=idx, zone="hand", hand_index=idx)
            for idx, card in enumerate(player.hand)
        ]

        playable_indexes: set[int] = set()
        legal_action_count = 0
        try:
            legal_actions = self._arena.get_legal_actions(user_id)
            legal_action_count = len(legal_actions)
            for action in legal_actions:
                if isinstance(action, PlayCardAction):
                    playable_indexes.add(int(action.hand_index))
        except Exception:
            legal_actions = []

        available_hand = [
            item for item in acting_hand
            if int(item.get("hand_index", -1)) in playable_indexes
        ]

        selected_card = None
        if str(action_json.get("type") or "") == "play_card":
            card_ref = action_json.get("card_ref", action_json.get("hand_index", 0))
            hand_index = self._resolve_hand_index(player.hand, card_ref)
            if 0 <= hand_index < len(player.hand):
                selected_card = self._card_params_slot_payload(
                    player.hand[hand_index],
                    slot=hand_index,
                    zone="hand",
                    hand_index=hand_index,
                )

        return {
            "schema": TRAIN_V3_ACTION_CONTEXT_SCHEMA,
            "card_params_schema": TRAIN_V3_CARD_PARAMS_SCHEMA,
            "deck_params_schema": TRAIN_V3_DECK_PARAMS_SCHEMA,
            "acting_player": acting_player,
            "game_mode": self.game_mode,
            "ruleset": self.ruleset,
            "turn_number": st.turn_number,
            "turn_start_mana": player.mana,
            "legal_action_count": legal_action_count,
            "acting_hand": acting_hand,
            "available_hand_cards": available_hand,
            "selected_card": selected_card,
            "deck_snapshot_ref": "battle_summary.metadata.deck_param_snapshots",
        }

    def record_analytics_action(
        self,
        user_id: int,
        action_json: dict[str, Any],
        quality_score: Optional[float] = None,
        *,
        decision_source: Optional[str] = None,
        control_source: Optional[str] = None,
        actor_type: Optional[str] = None,
        request_monotonic: Optional[float] = None,
        client_action_id: Optional[str] = None,
        human_decision_time_ms: Optional[float] = None,
        decision_time_censored: Optional[bool] = None,
        decision_censor_reason: Optional[str] = None,
        metronome_prediction_ms: Optional[float] = None,
        metronome_applied_ms: Optional[float] = None,
        metronome_fallback_used: Optional[bool] = None,
    ) -> Optional[dict[str, Any]]:
        if not self._arena:
            return None
        st = self._arena.state
        acting_player = 1 if user_id == st.p1.user_id else 2
        snapshot = self.build_analytics_snapshot()
        context = self._build_analytics_action_context(user_id, acting_player, action_json)
        legacy_action_index: Optional[int] = None
        if not self._analytics_flushed:
            legacy_action_index = len(self._analytics_actions)
            self._analytics_actions.append({
                "turn_number": st.turn_number,
                "acting_player": acting_player,
                "acting_user_id": user_id,
                "is_bot": self.is_bot(user_id),
                "state_json": snapshot or {},
                "action_json": action_json,
                "context_json": context,
                "quality_score": quality_score,
            })

        captured = self.capture_action_context(
            int(user_id),
            action_json,
            decision_source=decision_source,
            control_source=control_source,
            actor_type=actor_type,
            request_monotonic=request_monotonic,
            client_action_id=client_action_id,
            human_decision_time_ms=human_decision_time_ms,
            decision_time_censored=decision_time_censored,
            decision_censor_reason=decision_censor_reason,
            metronome_prediction_ms=metronome_prediction_ms,
            metronome_applied_ms=metronome_applied_ms,
            metronome_fallback_used=metronome_fallback_used,
            quality_score=quality_score,
            legacy_action_index=legacy_action_index,
            extra_context=context,
            queue=True,
        )
        # Preserve the legacy context schema while making the richer request
        # context visible to admin exports without exposing the private state.
        if legacy_action_index is not None:
            context["v5_action_context"] = {
                key: value
                for key, value in captured.items()
                if key
                not in {
                    "action_json",
                    "extra_context",
                    "legacy_action_index",
                    "request_monotonic_ms",
                }
            }
        return captured


class _LegacyPlayerStateWrapper:
    """Обертка для совместимости со старым кодом, обращающимся к p1_state/p2_state."""
    
    def __init__(self, core_state: CorePlayerState, name: str, avatar_url: Optional[str]):
        self._core = core_state
        self.name = name
        self.avatar_url = avatar_url
    
    @property
    def user_id(self) -> int:
        return self._core.user_id
    
    @property
    def hero_hp(self) -> int:
        return self._core.hero.hp
    
    @property
    def mana(self) -> int:
        return self._core.mana
    
    @property
    def max_mana(self) -> int:
        return self._core.max_mana
    
    @property
    def hand(self) -> List[CardInstance]:
        return self._core.hand
    
    @property
    def board(self) -> List[CardInstance]:
        return self._core.board
    
    @property
    def deck(self) -> List[CardInstance]:
        return self._core.deck
    
    @property
    def draw_pile(self) -> List[str]:
        # Совместимость: возвращаем ID карт в колоде
        return [str(c.card_id) for c in self._core.deck]
    
    @property
    def hero_attack(self) -> int:
        return self._core.hero.attack
    
    @property
    def hero_mechanics(self) -> List[str]:
        return self._core.hero.mechanics
    
    @property
    def hero_name(self) -> str:
        return self._core.hero.name
    
    @property
    def replacement_status(self) -> Any:
        """Статус замены игрока (ACTIVE/AFK/SURRENDERED)."""
        return self._core.replacement_status
    
    @replacement_status.setter
    def replacement_status(self, value: Any) -> None:
        """Устанавливает статус замены игрока."""
        self._core.replacement_status = value
    
    @property
    def surrender_processed(self) -> bool:
        """Флаг немедленного списания трофеев при сдаче."""
        return getattr(self._core, "surrender_processed", False)
    
    @surrender_processed.setter
    def surrender_processed(self, value: bool) -> None:
        """Устанавливает флаг surrender_processed."""
        self._core.surrender_processed = value


class _EmptyLegacyState:
    """Пустая заглушка для случаев, когда арена не инициализирована."""
    
    def __init__(self, user_id: int, name: str):
        from core.state import ReplacementStatus
        self.user_id = user_id
        self.name = name
        self.avatar_url = None
        self.hero_hp = 30
        self.mana = 0
        self.max_mana = 0
        self.hand = []
        self.board = []
        self.draw_pile = []
        self.hero_attack = 0
        self.hero_mechanics = []
        self.hero_name = "Герой"
        self.replacement_status = ReplacementStatus.ACTIVE
        self.surrender_processed = False
