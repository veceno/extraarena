"""OrchestraBattleEngine — контракт-шим над ``core.engine.ArenaEnvironment``.

Цель та же, что у ``rlhf_env/components/arena_engine.py``: воспроизвести
**байт-в-байт** сервер-контракт настоящей игры (``battle_engine.py``), чтобы
реальный ``webapp/arena.js`` (frozen-снапшот в ``webapp_borrow/``) рендерил
сцену без правок.

Отличия от ``RlhfBattleEngine``:
  - обе стороны описываются сценарием (нет жёсткого human/bot); ready-барьер
    отключён (сцена детерминирована, игра идёт по графам);
  - сериализаторы скопированы verbatim из ``battle_engine.py`` (включая
    ``mana_draw_count_this_turn``, которого нет в rlhf-шиме);
  - добавлен **порт sound_events** (``battle_engine.py:475-642``) с
    детерминированным ``event_id`` (``orchestra:<turn>:<kind>:<card>:<event>``)
    — arena.js дедуплит sfx по ``event_id``;
  - ``apply_action`` шагает ядро и возвращает снимок + sound_events одним
    вызовом (нужно scenario_engine'у для построения покадрового ролика).

Все действия проводятся через ``core.actions.{PlayCardAction,AttackAction,
EndTurnAction,ManaDrawAction}`` + ``engine.step`` — ровно как в проде.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

from core.actions import (
    AttackAction,
    BaseAction,
    EndTurnAction,
    ManaDrawAction,
    PlayCardAction,
)
from core.engine import ArenaEnvironment
from core.effects import get_taunt_targets, has_taunt
from core.state import (
    CardInstance,
    GameStatus,
    PlayerState as CorePlayerState,
    ReplacementStatus,
)
from infrastructure.card_assets import card_asset_url
from infrastructure.match_modes import (
    ClassicParams,
    ModeConfig,
    resolve_mode_config,
    serialize_mode_config,
)

logger = logging.getLogger(__name__)


def _profile_defaults(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    p = dict(profile or {})
    # В сценарии персона задаётся полем `nickname`; сериализатору нужно `name`.
    if not p.get("name") and p.get("nickname"):
        p["name"] = p["nickname"]
    p.setdefault("name", "")
    p.setdefault("avatar_url", None)
    p.setdefault("title", "")
    p.setdefault("rarity", "")
    p.setdefault("extra_pass", None)
    p.setdefault("background_url", None)
    p.setdefault("nickname_glow_disabled", False)
    p.setdefault("hide_player_id_public", False)
    p.setdefault("trophies", 0)
    p.setdefault("is_bot", False)
    return p


class OrchestraBattleEngine:
    """Контракт-шим над ``ArenaEnvironment`` для сценарного плеера 1:1."""

    def __init__(
        self,
        *,
        match_id: str,
        arena: ArenaEnvironment,
        mode_config: Optional[ModeConfig] = None,
        p1_user_id: int,
        p2_user_id: int,
        p1_profile: Optional[Dict[str, Any]] = None,
        p2_profile: Optional[Dict[str, Any]] = None,
        game_mode: str = "classic",
        event_id_prefix: str = "orchestra",
    ) -> None:
        self.match_id = match_id
        self._arena: ArenaEnvironment = arena
        self.mode_config: ModeConfig = mode_config or resolve_mode_config(game_mode)
        self.game_mode = self.mode_config.mode_id
        self.ruleset = self.mode_config.ruleset
        self.turn_duration = self.mode_config.classic.turn_duration_seconds

        self._event_id_prefix = event_id_prefix

        self.player_ids = [p1_user_id, p2_user_id]
        self.is_bot_match = False
        self.is_ended = False
        self.game_over_processed = False
        self.turn_start_monotonic: Optional[float] = None
        self.turn_time_history: List[Dict[str, Any]] = []
        self.client_ready = True
        # ready-барьер отключён: оба игрока «готовы» сразу (сцена идёт по графам).
        self.client_ready_users: set = {int(p1_user_id), int(p2_user_id)}

        p1p = _profile_defaults(p1_profile)
        p2p = _profile_defaults(p2_profile)
        self._p1_id = int(p1_user_id)
        self._p1_name = p1p["name"]
        self._p1_avatar_url = p1p["avatar_url"]
        self._p1_title = p1p["title"]
        self._p1_rarity = p1p["rarity"]
        self._p1_extra_pass = p1p["extra_pass"]
        self._p1_background_url = p1p["background_url"]
        self._p1_nickname_glow_disabled = bool(p1p["nickname_glow_disabled"])
        self._p1_hide_player_id_public = bool(p1p["hide_player_id_public"])

        self._p2_id = int(p2_user_id)
        self._p2_name = p2p["name"]
        self._p2_avatar_url = p2p["avatar_url"]
        self._p2_title = p2p["title"]
        self._p2_rarity = p2p["rarity"]
        self._p2_extra_pass = p2p["extra_pass"]
        self._p2_background_url = p2p["background_url"]
        self._p2_nickname_glow_disabled = bool(p2p["nickname_glow_disabled"])
        self._p2_hide_player_id_public = bool(p2p["hide_player_id_public"])

    # =========================================================================
    # ДЕЙСТВИЯ
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
        known_action = isinstance(action, (PlayCardAction, AttackAction, EndTurnAction, ManaDrawAction))
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

    def apply_action(self, user_id: int, action: BaseAction) -> Dict[str, Any]:
        """Шаг ядра + снимок + sound_events одним вызовом.

        Returns: ``{ok, error?, action_kind, snapshot, sound_events, game_over?,
        winner_id?}``.
        """
        if not self._arena:
            return {"ok": False, "error": "arena_not_initialized", "action_kind": action.__class__.__name__,
                    "snapshot": {}, "sound_events": []}
        preflight = self._preflight_action_error(user_id, action)
        if preflight is not None:
            return {"ok": False, "error": preflight,
                    "action_kind": _action_kind(action),
                    "snapshot": self.get_full_state(user_id), "sound_events": []}

        # Снапшот карты ДО действия (для sound_events) — пока карта ещё в руке/на поле.
        card_snapshot = self._sound_card_snapshot_for_action(user_id, action)
        turn_before = self._arena.state.turn_number

        state_snapshot = copy.deepcopy(self._arena.state)
        try:
            success, error = self._arena.step(user_id, action)
        except Exception as exc:  # noqa: BLE001
            self._arena.state = state_snapshot
            logger.error("apply_action failed: %s", exc, exc_info=True)
            return {"ok": False, "error": "action_failed",
                    "action_kind": _action_kind(action),
                    "snapshot": self.get_full_state(user_id), "sound_events": []}

        if not success:
            self._arena.state = state_snapshot
            return {"ok": False, "error": error,
                    "action_kind": _action_kind(action),
                    "snapshot": self.get_full_state(user_id), "sound_events": []}

        # feedback-events накапливаются в state во время step — забираем ПОСЛЕ.
        feedback_events = self._consume_card_feedback_events(self._arena.state)
        sound_events = self._sound_events_for_action(
            action=action,
            card_snapshot=card_snapshot,
            turn_number=turn_before,
            card_feedback_events=feedback_events,
        )

        state = self._arena.state
        result: Dict[str, Any] = {
            "ok": True,
            "action_kind": _action_kind(action),
            "snapshot": self.get_full_state(user_id),
            "sound_events": sound_events,
        }
        if state.status != GameStatus.ONGOING:
            self.is_ended = True
            winner_id = None
            if state.status == GameStatus.P1_WIN:
                winner_id = state.p1.user_id
            elif state.status == GameStatus.P2_WIN:
                winner_id = state.p2.user_id
            result["game_over"] = True
            result["winner_id"] = winner_id
        return result

    # удобные обёртки (для тестов / прямого вызова)
    def play_card(self, user_id: int, hand_index: int, target_id: Optional[str] = None,
                  position: Optional[int] = None, target_is_hero: bool = False) -> Dict[str, Any]:
        if target_is_hero and target_id is None:
            opponent = self._arena.state.p2 if self._arena.state.p1.user_id == user_id else self._arena.state.p1
            target_id = str(opponent.hero.instance_id)
        action = PlayCardAction(hand_index=hand_index, target_id=target_id, position=position)
        return self.apply_action(user_id, action)

    def attack_target(self, user_id: int, attacker_id: str, target_id: Optional[str] = None,
                      target_is_hero: bool = False) -> Dict[str, Any]:
        action = AttackAction(attacker_id=str(attacker_id),
                              target_id=None if target_is_hero else (str(target_id) if target_id is not None else None),
                              target_is_hero=target_is_hero)
        return self.apply_action(user_id, action)

    def end_turn(self, user_id: int) -> Dict[str, Any]:
        return self.apply_action(user_id, EndTurnAction())

    def mana_draw(self, user_id: int) -> Dict[str, Any]:
        return self.apply_action(user_id, ManaDrawAction())

    def get_legal_actions(self, user_id: int) -> List[Dict[str, Any]]:
        if not self._arena:
            return []
        return [self._serialize_action(a) for a in self._arena.get_legal_actions(user_id)]

    # =========================================================================
    # SOUND EVENTS (порт из battle_engine.py:475-642; event_id детерминированный)
    # =========================================================================

    @staticmethod
    def _sound_card_snapshot(card: Optional[CardInstance]) -> Optional[Dict[str, Any]]:
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
        self, user_id: int, action: BaseAction,
    ) -> Optional[Dict[str, Any]]:
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
            "aoe_freeze", "desk_freeze", "choose_shield_damage", "cast_random_spell",
            "consume_ally", "delete_target",
        }
        if value in active_exact:
            return True
        active_prefixes = (
            "battlecry_", "spell_", "aoe_damage_", "damage_", "heal_", "damage_all",
            "heal_all", "buff_all_", "summon_", "mana_gain_", "mana_drain_",
        )
        return value.startswith(active_prefixes)

    @staticmethod
    def _sound_event_id_part(value: Any) -> str:
        return str(value or "unknown").replace(":", "_")

    def _make_sound_event(
        self, *, turn_number: int, action_kind: str, card_snapshot: Dict[str, Any],
        event: str, mechanic: Optional[str] = None,
    ) -> Dict[str, Any]:
        instance_id = card_snapshot.get("instance_id")
        event_id_parts = [
            self._event_id_prefix,
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
    def _consume_card_feedback_events(state) -> List[Dict[str, Any]]:
        events = list(getattr(state, "pending_card_feedback_events", []) or [])
        if hasattr(state, "pending_card_feedback_events"):
            state.pending_card_feedback_events.clear()
        return events

    @staticmethod
    def _find_card_feedback_event(
        card_snapshot: Dict[str, Any], mechanic: Optional[str],
        feedback_events: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
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
        self, *, action: BaseAction, card_snapshot: Optional[Dict[str, Any]],
        turn_number: int, card_feedback_events: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if not card_snapshot:
            return []
        feedback_events = card_feedback_events or []
        if isinstance(action, PlayCardAction):
            events = [
                self._make_sound_event(turn_number=turn_number, action_kind="play_card",
                                       card_snapshot=card_snapshot, event="deploy")
            ]
            for mechanic in card_snapshot.get("mechanics", []):
                if self._is_play_sound_mechanic(mechanic):
                    event = self._make_sound_event(
                        turn_number=turn_number, action_kind="play_card",
                        card_snapshot=card_snapshot, event="mechanic", mechanic=mechanic,
                    )
                    feedback = self._find_card_feedback_event(card_snapshot, mechanic, feedback_events)
                    if feedback and feedback.get("effect_code"):
                        event["effect_code"] = feedback["effect_code"]
                    events.append(event)
            return events
        if isinstance(action, AttackAction):
            return [
                self._make_sound_event(turn_number=turn_number, action_kind="attack",
                                       card_snapshot=card_snapshot, event="attack")
            ]
        return []

    # =========================================================================
    # TIMER / READY HELPERS (сцена — без real-time таймера; ready-барьер снят)
    # =========================================================================

    def is_waiting_for_players(self) -> bool:
        return False

    def required_ready_user_ids(self) -> set:
        return set()

    def get_turn_time_remaining(self) -> float:
        return float(self.turn_duration)

    def _serialize_turn_time_history(self, viewer_id: Optional[int]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
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
    # СЕРИАЛИЗАЦИЯ (verbatim из battle_engine.py:1001-1297)
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
        }

    def _serialize_sudden_death_state(
        self, player_state: CorePlayerState, opponent_state: CorePlayerState,
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
        self, ps: CorePlayerState, name: str, avatar_url: Optional[str],
        title: str = "", rarity: str = "", extra_pass: Optional[str] = None,
        background_url: Optional[str] = None, show_hand: bool = True,
        nickname_glow_disabled: bool = False, hide_player_id_public: bool = False,
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
            "mana_draw_count_this_turn": ps.mana_draw_count_this_turn if show_hand else 0,
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
        self, action_history: List[tuple], viewer_id: Optional[int],
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


def _action_kind(action: BaseAction) -> str:
    if isinstance(action, PlayCardAction):
        return "play_card"
    if isinstance(action, AttackAction):
        return "attack"
    if isinstance(action, EndTurnAction):
        return "end_turn"
    if isinstance(action, ManaDrawAction):
        return "mana_draw"
    return action.__class__.__name__


__all__ = ["OrchestraBattleEngine"]