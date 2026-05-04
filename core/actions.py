"""
Пространство действий для пошагового движка (без логики боя).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from core.state import GameState


class BaseAction(ABC):
    """Базовое действие."""

    @abstractmethod
    def to_dict(self) -> Dict:
        ...

    @abstractmethod
    def validate(self, state: GameState) -> None:
        ...


@dataclass
class PlayCardAction(BaseAction):
    hand_index: int
    target_id: Optional[str] = None
    position: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "type": "play_card",
            "hand_index": self.hand_index,
            "target_id": self.target_id,
            "position": self.position,
        }

    def validate(self, state: GameState) -> None:
        if self.hand_index < 0:
            raise ValueError("hand_index должен быть неотрицательным")


@dataclass
class AttackAction(BaseAction):
    attacker_id: str
    target_id: Optional[str] = None
    target_is_hero: bool = False

    def to_dict(self) -> Dict:
        return {
            "type": "attack",
            "attacker_id": self.attacker_id,
            "target_id": self.target_id,
            "target_is_hero": self.target_is_hero,
        }

    def validate(self, state: GameState) -> None:
        if not self.attacker_id:
            raise ValueError("attacker_id обязателен")
        if not self.target_is_hero and not self.target_id:
            raise ValueError("target_id обязателен для атаки по существу")


@dataclass
class EndTurnAction(BaseAction):
    def to_dict(self) -> Dict:
        return {"type": "end_turn"}

    def validate(self, state: GameState) -> None:
        # Нет доп. проверок для завершения хода
        return
