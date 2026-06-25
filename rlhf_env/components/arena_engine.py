"""RlhfBattleEngine — тонкий контракт-шим над core.engine.ArenaEnvironment.

Цель: воспроизвести **байт-в-байт** сервервер-контракт настоящей игры
(web/server.py + battle_engine.py::BattleEngine), чтобы реальный
webapp/arena.js работал без модификаций в RLHF-среде.

От BattleEngine отличается тем, что:
  - не завязан на БД (cards_data строится из каталога ai/cards.json);
  - оборачивает уже созданный ArenaEnvironment (колоду/стейт строит
    ArenaMatchManager, а не create_match);
  - сериализаторы (get_full_state / _serialize_*) скопированы верbatim из
    battle_engine.py, чтобы wire-формат совпадал с продом;
  - не реализует talkies/sound_events/trophies/economy — arena.js терпит
    их отсутствие (пустые/минимальные значения).

Все действия проводятся через core.actions.{PlayCardAction,AttackAction,
EndTurnAction} + engine.step — ровно как в BattleEngine.execute_action.
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from core.actions import AttackAction, BaseAction, EndTurnAction, PlayCardAction
from core.engine import ArenaEnvironment
from core.effects import get_taunt_targets, has_taunt
from core.state import (
    CardInstance,
    GameStatus,
    PlayerState as CorePlayerState,
    ReplacementStatus,
)
from infrastructure.card_assets import card_asset_url
from infrastructure.match_modes import ModeConfig, resolve_mode_config, serialize_mode_config

logger = logging.getLogger(__name__)

# Schema-константы TrainV3 (скопированы из battle_engine.py:24-26 — единый
# источник правды для analytics.py и для deck_param_snapshot).
TRAIN_V3_CARD_PARAMS_SCHEMA = "train_v3_card_params_v1"
TRAIN_V3_ACTION_CONTEXT_SCHEMA = "train_v3_action_context_v1"
TRAIN_V3_DECK_PARAMS_SCHEMA = "train_v3_deck_params_v1"


class BattleEventEmitter:
    """Минимальный pub/sub для рассылки обновлений клиентам (копия из battle_engine.py:45)."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Any]] = {}

    def on(self, event_type: str, callback: Any) -> None:
        self._listeners.setdefault(event_type, []).append(callback)

    def emit(self, event_type: str, match_id: str, data: dict[str, Any]) -> None:
        for callback in self._listeners.get(event_type, ()):
            try:
                callback(match_id, data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Event callback failed: %s", exc)


class RlhfBattleEngine:
    """Контракт-шим над core.engine.ArenaEnvironment для RLHF-арены 1:1.

    human = p1 (user_id из профиля, по умолчанию 1000), бот = p2 (2000).
    Бот никогда не подключается по Socket.IO — его ходы разыгрывает
    match_runner.run_bot_turn через политику.
    """

    def __init__(
        self,
        *,
        match_id: str,
        arena: ArenaEnvironment,
        mode_config: Optional[ModeConfig] = None,
        human_user_id: int = 1000,
        bot_user_id: int = 2000,
        p1_profile: Optional[Dict[str, Any]] = None,
        p2_profile: Optional[Dict[str, Any]] = None,
        bot_difficulty: str = "max",
        bot_brain_profile: Optional[str] = None,
        game_mode: str = "classic",
        event_emitter: Optional[BattleEventEmitter] = None,
    ) -> None:
        self.match_id = match_id
        self._arena: ArenaEnvironment = arena
        self.mode_config: ModeConfig = mode_config or resolve_mode_config(game_mode)
        self.game_mode = self.mode_config.mode_id
        self.ruleset = self.mode_config.ruleset
        self.turn_duration = self.mode_config.classic.turn_duration_seconds

        self.human_user_id = human_user_id
        self.bot_user_id = bot_user_id
        self.player_ids = [human_user_id, bot_user_id]
        self.is_bot_match = True
        self.bot_id = bot_user_id
        self.bot_difficulty = bot_difficulty
        self.bot_difficulty_label = bot_difficulty
        self.bot_brain_profile = bot_brain_profile

        self.current_player_id: Optional[int] = None
        self.turn = 0
        self.is_ended = False
        self.game_over_processed = False
        self.turn_start_time: Optional[float] = None
        self.turn_start_monotonic: Optional[float] = None
        self.match_start_time: Optional[float] = None
        self.match_start_monotonic: Optional[float] = None
        self.turn_time_history: list[dict[str, Any]] = []
        self.client_ready = False
        self.client_ready_users: set[int] = set()

        self._event_emitter = event_emitter
        self._logger = logging.getLogger(__name__)

        p1p = p1_profile or {}
        p2p = p2_profile or {}
        # Профиль человека (p1)
        self._p1_id = human_user_id
        self._p1_name = p1p.get("name") or "Вы"
        self._p1_avatar_url = p1p.get("avatar_url")
        self._p1_trophies = int(p1p.get("trophies", 0) or 0)
        self._p1_clan = p1p.get("clan", "")
        self._p1_title = p1p.get("title", "")
        self._p1_rarity = p1p.get("rarity", "")
        self._p1_extra_pass = p1p.get("extra_pass")
        self._p1_background_url = p1p.get("background_url")
        self._p1_nickname_glow_disabled = bool(p1p.get("nickname_glow_disabled", False))
        self._p1_hide_player_id_public = bool(p1p.get("hide_player_id_public", False))
        # Профиль бота (p2)
        self._p2_id = bot_user_id
        self._p2_name = p2p.get("name") or f"Berserk {bot_difficulty}"
        self._p2_avatar_url = p2p.get("avatar_url")
        self._p2_trophies = int(p2p.get("trophies", 0) or 0)
        self._p2_clan = p2p.get("clan", "")
        self._p2_title = p2p.get("title", "")
        self._p2_rarity = p2p.get("rarity", "")
        self._p2_extra_pass = p2p.get("extra_pass")
        self._p2_background_url = p2p.get("background_url")
        self._p2_nickname_glow_disabled = bool(p2p.get("nickname_glow_disabled", False))
        self._p2_hide_player_id_public = bool(p2p.get("hide_player_id_public", False))

        # Исходные колоды для provenance/analytics (заполняет менеджер).
        self._p1_initial_deck_ids: list[int] = []
        self._p2_initial_deck_ids: list[int] = []
        self._p1_initial_deck_params: dict[str, Any] = {}
        self._p2_initial_deck_params: dict[str, Any] = {}
        # Источник колоды P1 для экрана превью (заполняет менеджер).
        self.p1_deck_source: dict[str, Any] = {"type": "random"}

        # Стартуем таймер первого хода сразу (бото-матч — без ready-барьера).
        if self._arena:
            self.current_player_id = self._arena.state.current_turn_owner_id
            self.turn = self._arena.state.turn_number
            self.turn_start_time = time.time()
            self.turn_start_monotonic = time.monotonic()
            self.match_start_time = self.turn_start_time
            self.match_start_monotonic = self.turn_start_monotonic

    # =========================================================================
    # СОЗДАНИЕ / ПРИВЯЗКА ДЕК
    # =========================================================================

    def set_initial_decks(
        self,
        p1_deck_ids: List[int],
        p2_deck_ids: List[int],
        p1_levels: Optional[Dict[int, int]] = None,
        p2_levels: Optional[Dict[int, int]] = None,
    ) -> None:
        """Запоминает исходные колоды для provenance и deck_param_snapshot."""
        self._p1_initial_deck_ids = list(p1_deck_ids)
        self._p2_initial_deck_ids = list(p2_deck_ids)
        # deck_param_snapshot строится лениво из CardInstance начальной колоды;
        # для analytics используем сами исходные карты из state (см. analytics.py).

    # =========================================================================
    # ДЕЙСТВИЯ (копия execute_action / play_card / attack_target / end_turn /
    # execute_bot_action / get_legal_actions / get_preview_delta из battle_engine.py)
    # =========================================================================

    def _preflight_action_error(self, user_id: int, action: BaseAction) -> Optional[str]:
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
        except Exception:  # noqa: BLE001
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

    def execute_action(self, user_id: int, action: BaseAction) -> Dict[str, Any]:
        if not self._arena:
            return {"success": False, "error": "arena_not_initialized"}
        preflight_error = self._preflight_action_error(user_id, action)
        if preflight_error is not None:
            return {"success": False, "error": preflight_error}

        state_snapshot = copy.deepcopy(self._arena.state)
        previous_turn_number = self._arena.state.turn_number
        previous_turn_owner_id = self._arena.state.current_turn_owner_id
        previous_turn_start_time = self.turn_start_time
        previous_turn_start_monotonic = self.turn_start_monotonic

        try:
            success, error = self._arena.step(user_id, action)
        except Exception as exc:  # noqa: BLE001
            self._arena.state = state_snapshot
            self.current_player_id = previous_turn_owner_id
            self.turn = previous_turn_number
            self.turn_start_time = previous_turn_start_time
            self.turn_start_monotonic = previous_turn_start_monotonic
            self._logger.error("execute_action failed: %s", exc, exc_info=True)
            return {"success": False, "error": "action_failed"}

        if not success:
            self._arena.state = state_snapshot
            self.current_player_id = previous_turn_owner_id
            self.turn = previous_turn_number
            self.turn_start_time = previous_turn_start_time
            self.turn_start_monotonic = previous_turn_start_monotonic
            return {"success": False, "error": error}

        state = self._arena.state
        self.current_player_id = state.current_turn_owner_id
        self.turn = state.turn_number

        if isinstance(action, EndTurnAction):
            self._record_completed_turn(
                previous_turn_number,
                previous_turn_owner_id,
                previous_turn_start_monotonic,
            )
            self.turn_start_time = time.time()
            self.turn_start_monotonic = time.monotonic()

        result: Dict[str, Any] = {"success": True}
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

        if self._event_emitter:
            self._event_emitter.emit(
                "state_changed",
                self.match_id,
                {
                    "event_type": "state_changed",
                    "match_id": self.match_id,
                    "action": action.to_dict(),
                    "actor_user_id": user_id,
                },
            )
        return result

    def _resolve_hand_index(self, hand: List[CardInstance], card_ref: Any) -> int:
        if isinstance(card_ref, int) and 0 <= card_ref < len(hand):
            return card_ref
        card_ref_str = str(card_ref)
        for i, card in enumerate(hand):
            if str(card.instance_id) == card_ref_str:
                return i
            if str(card.card_id) == card_ref_str:
                return i
        return -1

    def play_card(
        self,
        user_id: int,
        card_id_from_hand: Any,
        target_position: int,
        target_id: Optional[Any] = None,
        target_is_hero: bool = False,
    ) -> Dict[str, Any]:
        if not self._arena:
            return {"action": "play_card", "error": "arena_not_initialized"}
        state = self._arena.state
        player = state.p1 if state.p1.user_id == user_id else state.p2
        hand_index = self._resolve_hand_index(player.hand, card_id_from_hand)
        if hand_index < 0:
            return {"action": "play_card", "error": "card_not_found_in_hand"}
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
        if not self._arena:
            return {"action": "end_turn", "error": "arena_not_initialized"}
        action = EndTurnAction()
        result = self.execute_action(user_id, action)
        result["action"] = "end_turn"
        return result

    def execute_bot_action(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
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
        else:
            return {"error": f"unknown_action_type: {action_type}"}
        return self.execute_action(self.get_current_player_id(), action)

    def check_game_over(self) -> Dict[str, Any]:
        if not self._arena:
            return {"game_over": False, "winner_id": None}
        state = self._arena.state
        game_over = state.status != GameStatus.ONGOING
        winner_id = self._get_winner_id() if game_over else None
        if game_over:
            self.is_ended = True
        return {"game_over": game_over, "winner_id": winner_id}

    def get_legal_actions(self, user_id: int) -> List[Dict[str, Any]]:
        if not self._arena:
            return []
        actions = self._arena.get_legal_actions(user_id)
        return [self._serialize_action(a) for a in actions]

    def get_legal_actions_raw(self, user_id: int) -> List[BaseAction]:
        """Несериализованные BaseAction — для match_runner/бота."""
        if not self._arena:
            return []
        return self._arena.get_legal_actions(user_id)

    def get_preview_delta(self, action: BaseAction) -> Dict[str, int]:
        if not self._arena:
            return {}
        return self._arena.get_preview_delta(action)

    # =========================================================================
    # ТАЙМЕРЫ / HELPERS (копия из battle_engine.py)
    # =========================================================================

    def get_current_player_id(self) -> Optional[int]:
        if self._arena:
            return self._arena.state.current_turn_owner_id
        return self.current_player_id

    def get_turn_time_remaining(self) -> float:
        if self.is_waiting_for_players():
            return float(self.turn_duration)
        if self.turn_start_time is None:
            return float(self.turn_duration)
        start = self.turn_start_monotonic if self.turn_start_monotonic is not None else self.turn_start_time
        elapsed = time.monotonic() - start if self.turn_start_monotonic is not None else time.time() - start
        return max(0, self.turn_duration - elapsed)

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

    def is_bot(self, player_id: Optional[int]) -> bool:
        if not self._arena or player_id is None:
            return False
        state = self._arena.state
        if state.p1.user_id == player_id:
            return state.p1.is_bot
        if state.p2.user_id == player_id:
            return state.p2.is_bot
        return False

    def is_current_player_bot(self) -> bool:
        return self.is_bot(self.get_current_player_id())

    def mark_surrender(self, player_id: int) -> None:
        if not self._arena:
            return
        state = self._arena.state
        if state.p1.user_id == player_id:
            state.p1.replacement_status = ReplacementStatus.SURRENDERED
        elif state.p2.user_id == player_id:
            state.p2.replacement_status = ReplacementStatus.SURRENDERED

    def get_player_replacement_status(self, user_id: int) -> ReplacementStatus:
        if not self._arena:
            return ReplacementStatus.ACTIVE
        state = self._arena.state
        if state.p1.user_id == user_id:
            return getattr(state.p1, "replacement_status", ReplacementStatus.ACTIVE)
        if state.p2.user_id == user_id:
            return getattr(state.p2, "replacement_status", ReplacementStatus.ACTIVE)
        return ReplacementStatus.ACTIVE

    def restore_player_control(self, user_id: int) -> bool:
        status = self.get_player_replacement_status(user_id)
        if status == ReplacementStatus.SURRENDERED:
            return False
        if status == ReplacementStatus.AFK:
            self._set_replacement_status(user_id, ReplacementStatus.ACTIVE)
        return status == ReplacementStatus.AFK

    def mark_player_activity(self, user_id: int) -> None:
        self.restore_player_control(user_id)

    def _set_replacement_status(self, user_id: int, status: ReplacementStatus) -> None:
        if not self._arena:
            return
        state = self._arena.state
        if state.p1.user_id == user_id:
            state.p1.replacement_status = status
        elif state.p2.user_id == user_id:
            state.p2.replacement_status = status

    def required_ready_user_ids(self) -> set[int]:
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
                    required.add(int(player.user_id))
                except (TypeError, ValueError):
                    continue
        return required

    def _requires_ready_barrier(self) -> bool:
        # Бото-матч — барьер готовности не нужен; человек один.
        if self.is_bot_match or not self._arena:
            return False
        return len(self.required_ready_user_ids()) >= 2

    def is_waiting_for_players(self) -> bool:
        if self.is_ended or not self._requires_ready_barrier():
            return False
        return not self.required_ready_user_ids().issubset(self.client_ready_users)

    def mark_client_ready(self, user_id: Optional[int] = None) -> dict[str, Any]:
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
                self.match_start_time = self.match_start_time or self.turn_start_time
                self.match_start_monotonic = self.match_start_monotonic or self.turn_start_monotonic
            self.client_ready = True
        else:
            self.client_ready = False
        ready_count = len(self.client_ready_users.intersection(required_ids))
        return {
            "all_ready": bool(self.client_ready),
            "ready_user_ids": sorted(self.client_ready_users),
            "waiting_for_user_ids": sorted(required_ids - self.client_ready_users),
            "ready_count": ready_count,
            "required_ready_count": len(required_ids),
        }

    def _get_winner_id(self) -> Optional[int]:
        if not self._arena:
            return None
        status = self._arena.state.status
        if status == GameStatus.P1_WIN:
            return self._arena.state.p1.user_id
        if status == GameStatus.P2_WIN:
            return self._arena.state.p2.user_id
        return None

    # =========================================================================
    # СЕРИАЛИЗАЦИЯ (верbatim из battle_engine.py:1056-1281, 1103-1142)
    # =========================================================================

    def get_full_state(self, viewer_id: Optional[int] = None) -> Dict[str, Any]:
        if not self._arena:
            return {"match_id": self.match_id, "error": "arena_not_initialized"}

        state = self._arena.state
        p1 = state.p1
        p2 = state.p2

        viewer_is_p1 = viewer_id is not None and p1.user_id == viewer_id
        viewer_is_p2 = viewer_id is not None and p2.user_id == viewer_id
        known_viewer = viewer_is_p1 or viewer_is_p2

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

        legal_actions_json: List[Dict[str, Any]] = []
        if known_viewer and is_my_turn:
            legal_actions = self._arena.get_legal_actions(viewer_id)
            legal_actions_json = [self._serialize_action(a) for a in legal_actions]

        required_ready_user_ids = self.required_ready_user_ids()
        waiting_for_user_ids = [
            user_id for user_id in required_ready_user_ids if user_id not in self.client_ready_users
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
            "player1_hp": p1.hero.hp,
            "player2_hp": p2.hero.hp,
            "player": self._serialize_player_state(
                player_state, player_name, player_avatar, player_title, player_rarity,
                player_extra_pass, player_background, show_hand=known_viewer,
                nickname_glow_disabled=player_glow_disabled,
                hide_player_id_public=player_hide_id,
            ),
            "opponent": self._serialize_player_state(
                opponent_state, opponent_name, opponent_avatar, opponent_title, opponent_rarity,
                opponent_extra_pass, opponent_background, show_hand=False,
                nickname_glow_disabled=opponent_glow_disabled,
                hide_player_id_public=opponent_hide_id,
            ),
            "legal_actions": legal_actions_json,
            "action_history": self._serialize_action_history(state.action_history, viewer_id),
            # Источник колоды P1 + исходные card_ids — для предбоевого экрана превью
            # (только случайная колода; импортированная превью не требует).
            "p1_deck_source": dict(self.p1_deck_source),
            "p1_deck_card_ids": list(self._p1_initial_deck_ids),
        }

    def _serialize_sudden_death_state(
        self,
        player_state: CorePlayerState,
        opponent_state: CorePlayerState,
    ) -> Dict[str, Any]:
        classic = self.mode_config.classic
        if not classic.sudden_death_enabled or not self._arena:
            return {"enabled": False}
        state = self._arena.state

        def _turn_count(ps: CorePlayerState) -> int:
            return int(state.sudden_death_turns_by_player.get(int(ps.user_id), 0))

        def _next_damage(turn_count: int) -> int:
            return classic.sudden_death_damage_start + turn_count * classic.sudden_death_damage_step

        def _turn_damage(ps: CorePlayerState, turn_count: int) -> Optional[int]:
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
        hand_data: List[Any] = []
        if show_hand:
            hand_data = [self._serialize_card(c) for c in ps.hand]
        else:
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
            "trophies": ps.trophies,
            "hero": self._serialize_card(ps.hero),
            "hand": hand_data,
            "hand_count": len(ps.hand),
            "deck_count": len(ps.deck),
            "board": [self._serialize_card(c, owner_id=ps.user_id) for c in ps.board],
        }

    def _serialize_card(self, card: CardInstance, owner_id: Optional[int] = None) -> Dict[str, Any]:
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
        data = action.to_dict()
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
        viewer_id: Optional[int],
    ) -> List[Dict[str, str]]:
        result = []
        for log_type, text in action_history:
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
            if prefix:
                if ":" in text and not text.startswith(prefix):
                    parts = text.split(":", 1)
                    final_text = f"{prefix} {parts[0]}:{parts[1]}"
                else:
                    final_text = f"{prefix} {text}"
            else:
                final_text = text
            result.append({"type": final_type, "text": final_text})
        return result

    # =========================================================================
    # ANALYTICS-HELPERS (копия из battle_engine.py:1698-1764) — используются
    # analytics.py для deck_param_snapshots и card_params.
    # =========================================================================

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
    def _deck_param_snapshot(cls, cards: List[CardInstance]) -> dict[str, Any]:
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
        card_params = RlhfBattleEngine._card_params_payload(card)
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


__all__ = ["RlhfBattleEngine", "BattleEventEmitter"]