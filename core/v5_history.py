"""Serializable action-history events for the Extra-LR V5 observation tape.

The battle core deliberately does not import anything from ``ai``.  This module
keeps the production-only event contract next to the state transition that
produces it.  Native ``GameState.history`` remains untouched; these events live
in the separate 20-slot ``GameState.v5_history_events`` ring used by the Rust
Block-B trainer, Phase-C replay, and live V5 inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.actions import (
    AttackAction,
    BaseAction,
    EndTurnAction,
    ManaDrawAction,
    PlayCardAction,
)
from core.state import CardInstance, CardType, GameState, PlayerState


_PLAY_BASE = 1
_PLAY_STRIDE = 8 * 17
_ATTACK_BASE = _PLAY_BASE + 4 * _PLAY_STRIDE


@dataclass(frozen=True)
class _OutcomeSnapshot:
    own_hero_hp: int
    enemy_hero_hp: int
    own_board_count: int
    enemy_board_count: int
    own_board_power: float
    enemy_board_power: float


@dataclass(frozen=True)
class V5HistoryCapture:
    actor_id: int
    action_id: int
    action_type: str
    source_card: dict[str, Any] | None
    target_card: dict[str, Any] | None
    pre: _OutcomeSnapshot


def capture_v5_history_event(
    actor_id: int,
    player: PlayerState,
    opponent: PlayerState,
    action: BaseAction,
) -> V5HistoryCapture:
    """Capture action identity and the actor-relative pre-transition state."""

    source_card: CardInstance | None = None
    target_card: CardInstance | None = None

    if isinstance(action, PlayCardAction):
        if 0 <= action.hand_index < len(player.hand):
            source_card = player.hand[action.hand_index]
        target_card = _find_target_card(player, opponent, action.target_id)
    elif isinstance(action, AttackAction):
        source_card = _find_card(player.board, action.attacker_id)
        target_card = (
            opponent.hero
            if action.target_is_hero
            else _find_card(opponent.board, action.target_id)
        )

    return V5HistoryCapture(
        actor_id=int(actor_id),
        action_id=_action_id(player, opponent, action),
        action_type=str(action.to_dict().get("type") or "unknown"),
        source_card=snapshot_v5_history_card(source_card),
        target_card=snapshot_v5_history_card(target_card),
        pre=_outcome_snapshot(player, opponent),
    )


def finalize_v5_history_event(
    state: GameState,
    player: PlayerState,
    opponent: PlayerState,
    capture: V5HistoryCapture,
) -> dict[str, Any]:
    """Return the rich event after the accepted transition has settled.

    Source/target shapes come from ``capture`` (PRE action); deltas and the
    turn number come from the settled POST action state.
    """

    post = _outcome_snapshot(player, opponent)
    pre_board_delta = capture.pre.own_board_power - capture.pre.enemy_board_power
    post_board_delta = post.own_board_power - post.enemy_board_power

    return {
        "actor_id": capture.actor_id,
        "action_id": capture.action_id,
        "action_type": capture.action_type,
        "enemy_hero_hp_delta": capture.pre.enemy_hero_hp - post.enemy_hero_hp,
        "own_hero_hp_delta": capture.pre.own_hero_hp - post.own_hero_hp,
        "my_board_count_delta": post.own_board_count - capture.pre.own_board_count,
        "enemy_board_count_delta": post.enemy_board_count - capture.pre.enemy_board_count,
        "board_power_delta": float(post_board_delta - pre_board_delta),
        # TrainV3 records the post-transition turn number.  In particular, an
        # end_turn event belongs to the newly opened global turn.
        "turn_number": int(state.turn_number),
        "source_card": capture.source_card,
        "target_card": capture.target_card,
    }


def _outcome_snapshot(player: PlayerState, opponent: PlayerState) -> _OutcomeSnapshot:
    return _OutcomeSnapshot(
        own_hero_hp=int(player.hero.hp),
        enemy_hero_hp=int(opponent.hero.hp),
        own_board_count=len(player.board),
        enemy_board_count=len(opponent.board),
        own_board_power=_board_power(player.board),
        enemy_board_power=_board_power(opponent.board),
    )


def _board_power(board: list[CardInstance]) -> float:
    return float(
        sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in board)
    )


def snapshot_v5_history_card(card: CardInstance | None) -> dict[str, Any] | None:
    """Freeze the PRE-action card shape in the persisted Phase-C schema."""

    if card is None:
        return None
    card_type = (
        card.card_type.value
        if isinstance(card.card_type, CardType)
        else str(card.card_type)
    )
    mana_cost = int(card.mana_cost)
    attack = int(card.attack)
    mechanics = [str(value) for value in card.mechanics]
    return {
        "instance_id": str(card.instance_id),
        "id": int(card.card_id),
        "card_id": int(card.card_id),
        "name": str(card.name),
        "type": card_type,
        "card_type": card_type,
        "rarity": str(card.rarity),
        "level": int(card.level),
        "mana": mana_cost,
        "mana_cost": mana_cost,
        "atk": attack,
        "attack": attack,
        "hp": int(card.hp),
        "max_hp": int(card.max_hp),
        "is_ready": bool(card.is_ready),
        "is_frozen": bool(card.is_frozen),
        "mechanics": mechanics,
        "card_params": {
            "schema": "train_v3_card_params_v1",
            "type": card_type,
            "mana_cost": mana_cost,
            "attack": attack,
            "hp": int(card.hp),
            "max_hp": int(card.max_hp),
            "mechanics": mechanics,
            "is_ready": bool(card.is_ready),
            "is_frozen": bool(card.is_frozen),
            "level": int(card.level),
        },
    }


def _find_card(cards: list[CardInstance], instance_id: Any) -> CardInstance | None:
    if instance_id is None:
        return None
    target = str(instance_id)
    for card in cards:
        if str(card.instance_id) == target:
            return card
    return None


def _find_target_card(
    player: PlayerState,
    opponent: PlayerState,
    instance_id: Any,
) -> CardInstance | None:
    return _find_card(
        [player.hero, opponent.hero, *player.board, *opponent.board],
        instance_id,
    )


def _action_id(
    player: PlayerState,
    opponent: PlayerState,
    action: BaseAction,
) -> int:
    """Mirror the frozen 601-candidate codec without importing ``ai``."""

    if isinstance(action, EndTurnAction):
        return 0
    if isinstance(action, ManaDrawAction):
        # Mana draw is a parallel binary head and has no 601-candidate id.
        return 0
    if isinstance(action, PlayCardAction):
        hand_index = int(action.hand_index)
        position = int(action.position or 0)
        target_code = _play_target_code(player, opponent, action.target_id)
        return _PLAY_BASE + hand_index * _PLAY_STRIDE + position * 17 + target_code
    if isinstance(action, AttackAction):
        attacker_index = _card_index(player.board, action.attacker_id)
        if attacker_index is None:
            return 0
        target_code = (
            7
            if action.target_is_hero
            else _enemy_board_index(opponent, action.target_id)
        )
        if target_code is None:
            return 0
        return _ATTACK_BASE + min(attacker_index, 6) * 8 + target_code
    return 0


def _play_target_code(
    player: PlayerState,
    opponent: PlayerState,
    target_id: Any,
) -> int:
    if target_id is None:
        return 0
    target = str(target_id)
    if str(opponent.hero.instance_id) == target:
        return 8
    if str(player.hero.instance_id) == target:
        return 16
    enemy_index = _card_index(opponent.board, target)
    if enemy_index is not None and enemy_index <= 6:
        return 1 + enemy_index
    own_index = _card_index(player.board, target)
    if own_index is not None and own_index <= 6:
        return 9 + own_index
    return 0


def _enemy_board_index(opponent: PlayerState, target_id: Any) -> int | None:
    index = _card_index(opponent.board, target_id)
    return index if index is not None and index <= 6 else None


def _card_index(cards: list[CardInstance], instance_id: Any) -> int | None:
    if instance_id is None:
        return None
    target = str(instance_id)
    for index, card in enumerate(cards):
        if str(card.instance_id) == target:
            return index
    return None


__all__ = [
    "V5HistoryCapture",
    "capture_v5_history_event",
    "finalize_v5_history_event",
    "snapshot_v5_history_card",
]
