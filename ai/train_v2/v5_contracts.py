"""TrainV3/V5 environment tensor contracts.

These constants are training-only. They deliberately live outside production bot
runtime so V5 experiments can evolve without changing `ai.bot_brain`.

All V5 dims are DERIVED from ``CARD_SHAPE_DIM_V5`` (named offsets, not literals)
so the dim-73 cascade stays consistent when the card shape grows.
``OBS_V1_DIM`` (1456) is frozen — it backs the V4-orig ONNX warm-start.
"""
from __future__ import annotations

from dataclasses import dataclass

from ai.train_v2.classic_actions_v1 import ACTION_FEATURE_DIM, MAX_CANDIDATE_ACTIONS
from ai.train_v2.classic_card_shape_v1 import CARD_SHAPE_DIM  # frozen 64 (re-exported for classic consumers)
from ai.train_v2.classic_obs_v1 import OBS_DIM as OBS_V1_DIM
from ai.train_v2.v5_card_shape_v1 import CARD_SHAPE_DIM_V5

OWN_HAND_SLOTS = 4
OWN_DECK_SLOTS = 12
ENEMY_HAND_SLOTS = 4
ENEMY_DECK_SLOTS = 12

# Per-card private-info slot = [occupied_flag, card_id_norm, card_shape_v5(73)].
PRIVATE_CARD_SLOT_DIM = 1 + 1 + CARD_SHAPE_DIM_V5  # 75
PRIVATE_CARD_SLOTS = OWN_HAND_SLOTS + OWN_DECK_SLOTS + ENEMY_HAND_SLOTS + ENEMY_DECK_SLOTS  # 32
PRIVATE_INFO_DIM = PRIVATE_CARD_SLOTS * PRIVATE_CARD_SLOT_DIM  # 32 * 75 = 2400

V5_GLOBAL_DIM = 32
HISTORY_EVENTS = 20
# History event = [13 metadata + 3 padding][source_card_v5][target_card_v5].
HISTORY_EVENT_SOURCE_OFFSET = 16
HISTORY_EVENT_DIM = HISTORY_EVENT_SOURCE_OFFSET + CARD_SHAPE_DIM_V5 * 2  # 16 + 146 = 162
HISTORY_DIM = HISTORY_EVENTS * HISTORY_EVENT_DIM  # 20 * 162 = 3240

OBS_V5_DIM = OBS_V1_DIM + V5_GLOBAL_DIM + PRIVATE_INFO_DIM + HISTORY_DIM  # 1456 + 32 + 2400 + 3240 = 7128


@dataclass(frozen=True)
class InfoModeV5:
    """Information visibility flags for an omniscient V5 observation."""

    adaptive_strength: float = 1.0
    own_hand_identity_known: bool = True
    own_deck_known: bool = True
    enemy_hand_known: bool = True
    enemy_deck_known: bool = True
    enemy_deck_order_known: bool = True
    draw_assist_enabled: bool = False
    draw_assist_strength: float = 0.0

    def clipped_strength(self) -> float:
        return max(0.0, min(1.0, float(self.adaptive_strength)))

    def clipped_draw_assist_strength(self) -> float:
        if not self.draw_assist_enabled:
            return 0.0
        return max(0.0, min(1.0, float(self.draw_assist_strength)))

    def has_private_info(self) -> bool:
        return any(
            (
                self.own_hand_identity_known,
                self.own_deck_known,
                self.enemy_hand_known,
                self.enemy_deck_known,
                self.enemy_deck_order_known,
            )
        )


@dataclass(frozen=True)
class AssistModeV5:
    """External training/production assist channels independent from AdaptiveStrength."""

    assembler_enabled: bool = False
    assembler_strength: float = 0.0
    desirerer_enabled: bool = False
    desirerer_strength: float = 0.0
    teacher_hint_available: bool = False
    assist_profile_id: int = 0

    def clipped_assembler_strength(self) -> float:
        if not self.assembler_enabled:
            return 0.0
        return max(0.0, min(1.0, float(self.assembler_strength)))

    def clipped_desirerer_strength(self) -> float:
        if not self.desirerer_enabled:
            return 0.0
        return max(0.0, min(1.0, float(self.desirerer_strength)))

    def clipped_profile_id(self) -> int:
        return max(0, min(15, int(self.assist_profile_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "assembler_enabled": bool(self.assembler_enabled),
            "assembler_strength": float(self.clipped_assembler_strength()),
            "desirerer_enabled": bool(self.desirerer_enabled),
            "desirerer_strength": float(self.clipped_desirerer_strength()),
            "teacher_hint_available": bool(self.teacher_hint_available),
            "assist_profile_id": int(self.clipped_profile_id()),
        }


__all__ = [
    "ACTION_FEATURE_DIM",
    "AssistModeV5",
    "CARD_SHAPE_DIM",
    "CARD_SHAPE_DIM_V5",
    "ENEMY_DECK_SLOTS",
    "ENEMY_HAND_SLOTS",
    "HISTORY_DIM",
    "HISTORY_EVENT_DIM",
    "HISTORY_EVENT_SOURCE_OFFSET",
    "HISTORY_EVENTS",
    "InfoModeV5",
    "MAX_CANDIDATE_ACTIONS",
    "OBS_V1_DIM",
    "OBS_V5_DIM",
    "OWN_DECK_SLOTS",
    "OWN_HAND_SLOTS",
    "PRIVATE_CARD_SLOT_DIM",
    "PRIVATE_CARD_SLOTS",
    "PRIVATE_INFO_DIM",
    "V5_GLOBAL_DIM",
]
