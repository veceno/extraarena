"""
Core game engine package.
Содержит чистую синхронную логику боя для RL-агентов.
"""

from core.actions import AttackAction, BaseAction, EndTurnAction, PlayCardAction
from core.converter import card_from_db, deck_from_card_ids
from core.effects import process_effects, requires_target
from core.engine import ArenaEnvironment
from core.state import (
    CardInstance,
    CardType,
    GameState,
    GameStatus,
    PlayerState,
)

__all__ = [
    # Actions
    "BaseAction",
    "PlayCardAction",
    "AttackAction",
    "EndTurnAction",
    # State
    "CardInstance",
    "CardType",
    "GameState",
    "GameStatus",
    "PlayerState",
    # Engine
    "ArenaEnvironment",
    # Effects
    "process_effects",
    "requires_target",
    # Converter
    "card_from_db",
    "deck_from_card_ids",
]


