"""
Fast action cache for ClassicRLEnv — computes mask+legal_ids+features once per state.
Invalidated on every successful env.step().
"""
from __future__ import annotations

import numpy as np

from core.state import GameState

from ai.train_v2.classic_actions_v1 import (
    build_action_mask,
    encode_action_features,
    MAX_CANDIDATE_ACTIONS,
    ACTION_FEATURE_DIM,
)


class ActionCache:
    """
    Cache for a single GameState snapshot.
    Computes mask, legal action ids, and action features lazily,
    then serves them from memory until invalidated.
    """

    __slots__ = ("_state", "_player_id", "_mask", "_legal_ids", "_features",
                 "_include_preview", "_verify_mask", "_placement_mode",
                 "_mask_valid", "_features_valid")

    def __init__(
        self,
        state: GameState,
        player_id: int,
        *,
        verify_mask: bool = True,
        placement_mode: str = "full",
    ):
        self._state = state
        self._player_id = player_id
        self._mask: np.ndarray | None = None
        self._legal_ids: list[int] | None = None
        self._features: np.ndarray | None = None
        self._include_preview: bool = True
        self._verify_mask = verify_mask
        self._placement_mode = placement_mode
        self._mask_valid = False
        self._features_valid = False

    def invalidate(self) -> None:
        """Drop all cached data. Called after any successful env step."""
        self._mask = None
        self._legal_ids = None
        self._features = None
        self._mask_valid = False
        self._features_valid = False

    def set_state(self, state: GameState, player_id: int) -> None:
        """Rebind to a new state snapshot (used by env after step)."""
        self._state = state
        self._player_id = player_id
        self.invalidate()

    def mask(self) -> np.ndarray:
        if not self._mask_valid or self._mask is None:
            self._mask = build_action_mask(
                self._state, self._player_id,
                verify_mask=self._verify_mask,
                placement_mode=self._placement_mode,
            )
            self._mask_valid = True
        return self._mask

    def legal_ids(self) -> list[int]:
        if self._legal_ids is None:
            m = self.mask()
            self._legal_ids = [int(i) for i in range(MAX_CANDIDATE_ACTIONS) if m[i] == 1.0]
        return self._legal_ids

    def features(self, *, include_preview: bool = True) -> np.ndarray:
        if (
            not self._features_valid
            or self._features is None
            or self._include_preview != include_preview
        ):
            self._features = encode_action_features(
                self._state, self._player_id,
                include_preview=include_preview,
                verify_mask=self._verify_mask,
                placement_mode=self._placement_mode,
                mask=self.mask(),
            )
            self._include_preview = include_preview
            self._features_valid = True
        return self._features

    def reset(self, state: GameState, player_id: int) -> None:
        self._state = state
        self._player_id = player_id
        self.invalidate()
