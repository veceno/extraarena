"""Runtime adapters for the learned ExtraLR V1 auxiliary models.

The training script intentionally writes small, framework-independent ridge
artifacts.  This module is the matching inference contract used by benchmarks
and, later, production integration.  Feature construction is kept byte-for-byte
compatible with ``TrainV3.5/scripts/train_aux_models_v1.py``.
"""
from __future__ import annotations

import json
import math
import random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CARD_CATALOG = {
    int(card["id"]): card
    for card in json.loads((ROOT / "ai" / "cards.json").read_text(encoding="utf-8"))
}
CARD_IDS = tuple(sorted(CARD_CATALOG))
CARD_INDEX = {card_id: index for index, card_id in enumerate(CARD_IDS)}
HERO_CARD_IDS = frozenset(
    card_id
    for card_id, card in CARD_CATALOG.items()
    if card.get("card_type") == "hero"
)
ACTION_TYPES = ("attack", "play_card", "mana_draw", "end_turn")


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    with np.load(artifact, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def _ridge_predict(
    model: dict[str, np.ndarray],
    features: np.ndarray,
    *,
    prefix: str = "",
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    mean = model[f"{prefix}feature_mean"].astype(np.float64)
    scale = model[f"{prefix}feature_scale"].astype(np.float64)
    coef = model[f"{prefix}coef"].astype(np.float64)
    intercept = float(model[f"{prefix}intercept"][0])
    if x.shape[-1] != mean.shape[0]:
        raise ValueError(
            f"feature dimension mismatch: got {x.shape[-1]}, expected {mean.shape[0]}"
        )
    return ((x - mean) / scale) @ coef + intercept


def deck_vector(
    deck_ids: Iterable[int],
    levels: dict[str, Any] | dict[int, Any] | None = None,
) -> np.ndarray:
    out = np.zeros(len(CARD_IDS) * 2, dtype=np.float64)
    levels = levels or {}
    for raw_card_id in deck_ids:
        card_id = int(raw_card_id)
        index = CARD_INDEX.get(card_id)
        if index is None:
            continue
        out[index] += 1.0
        level = levels.get(str(card_id), levels.get(card_id, 1))
        out[len(CARD_IDS) + index] = max(
            out[len(CARD_IDS) + index],
            float(level) / 10.0,
        )
    return out


def pool_vector(card_ids: Iterable[int]) -> np.ndarray:
    out = np.zeros(len(CARD_IDS), dtype=np.float64)
    for raw_card_id in card_ids:
        index = CARD_INDEX.get(int(raw_card_id))
        if index is not None:
            out[index] = 1.0
    return out


def card_snapshot(card: Any) -> dict[str, Any]:
    return {
        "card_id": int(getattr(card, "card_id", 0) or 0),
        "level": int(getattr(card, "level", 1) or 1),
        "attack": int(getattr(card, "attack", 0) or 0),
        "hp": int(getattr(card, "hp", 0) or 0),
        "max_hp": int(getattr(card, "max_hp", 0) or 0),
        "mana_cost": int(getattr(card, "mana_cost", 0) or 0),
        "ready": bool(getattr(card, "is_ready", False)),
        "frozen": bool(getattr(card, "is_frozen", False)),
        "skip_count": int(getattr(card, "skip_count", 0) or 0),
    }


def visible_state(state: Any, actor_id: int) -> dict[str, Any]:
    actor = state.p1 if int(state.p1.user_id) == int(actor_id) else state.p2
    opponent = state.p2 if actor is state.p1 else state.p1

    def public_player(player: Any, *, private: bool) -> dict[str, Any]:
        payload = {
            "hero": card_snapshot(player.hero),
            "mana": int(player.mana),
            "max_mana": int(player.max_mana),
            "mana_draw_count_this_turn": int(player.mana_draw_count_this_turn),
            "hand_count": len(player.hand),
            "deck_count": len(player.deck),
            "board": [card_snapshot(card) for card in player.board],
        }
        if private:
            payload.update(
                {
                    "hand": [card_snapshot(card) for card in player.hand],
                    "remaining_deck": [card_snapshot(card) for card in player.deck],
                    "graveyard": [card_snapshot(card) for card in player.graveyard],
                }
            )
        return payload

    return {
        "turn_number": int(state.turn_number),
        "actor_id": int(actor_id),
        "information_mode": "actor_private_opponent_public_v1",
        "actor": public_player(actor, private=True),
        "opponent": public_player(opponent, private=False),
    }


def trace_visible_state(
    state: dict[str, Any],
    actor_player: int,
) -> dict[str, Any]:
    """Project an omniscient V5 trace state to the runtime human view.

    Stored V5 actions use ``p1``/``p2`` while runtime inference uses
    ``actor``/``opponent``.  Keeping this conversion in the inference module
    prevents training and serving from silently constructing different
    Metronome features.
    """

    actor_player = int(actor_player)
    if actor_player not in (1, 2):
        raise ValueError(f"actor_player must be 1 or 2, got {actor_player}")
    actor = state.get(f"p{actor_player}") or {}
    opponent = state.get(f"p{3 - actor_player}") or {}

    def trace_card(card: dict[str, Any]) -> dict[str, Any]:
        return {
            "card_id": int(card.get("card_id", card.get("id", 0)) or 0),
            "level": int(card.get("level", 1) or 1),
            "attack": int(card.get("attack", card.get("atk", 0)) or 0),
            "hp": int(card.get("hp", 0) or 0),
            "max_hp": int(card.get("max_hp", card.get("hp", 0)) or 0),
            "mana_cost": int(card.get("mana_cost", card.get("mana", 0)) or 0),
            "ready": bool(card.get("ready", card.get("is_ready", False))),
            "frozen": bool(card.get("frozen", card.get("is_frozen", False))),
            "skip_count": int(card.get("skip_count", 0) or 0),
        }

    def public_player(
        player: dict[str, Any],
        *,
        private: bool,
    ) -> dict[str, Any]:
        hand = player.get("hand") or []
        deck = player.get("deck") or []
        payload = {
            "hero": trace_card(player.get("hero") or {}),
            "mana": int(player.get("mana", 0) or 0),
            "max_mana": int(player.get("max_mana", 0) or 0),
            "mana_draw_count_this_turn": int(
                player.get("mana_draw_count_this_turn", 0) or 0
            ),
            "hand_count": len(hand),
            "deck_count": len(deck),
            "board": [trace_card(card) for card in player.get("board") or []],
        }
        if private:
            payload.update(
                {
                    "hand": [trace_card(card) for card in hand],
                    "remaining_deck": [trace_card(card) for card in deck],
                    "graveyard": [
                        trace_card(card) for card in player.get("graveyard") or []
                    ],
                }
            )
        return payload

    return {
        "turn_number": int(state.get("turn_number", 0) or 0),
        "actor_player": actor_player,
        "information_mode": "actor_private_opponent_public_v1",
        "actor": public_player(actor, private=True),
        "opponent": public_player(opponent, private=False),
    }


def state_scalars(state: dict[str, Any]) -> np.ndarray:
    actor = state.get("actor") or {}
    opponent = state.get("opponent") or {}

    def board_stats(player: dict[str, Any]) -> tuple[float, float, float, float]:
        board = player.get("board") or []
        return (
            len(board) / 5.0,
            sum(float(card.get("attack", 0)) for card in board) / 50.0,
            sum(float(card.get("hp", 0)) for card in board) / 50.0,
            sum(bool(card.get("ready")) for card in board) / 5.0,
        )

    actor_hero = actor.get("hero") or {}
    opponent_hero = opponent.get("hero") or {}
    return np.asarray(
        [
            math.log1p(float(state.get("turn_number", 0))) / 4.0,
            float(actor_hero.get("hp", 0))
            / max(float(actor_hero.get("max_hp", 1)), 1.0),
            float(opponent_hero.get("hp", 0))
            / max(float(opponent_hero.get("max_hp", 1)), 1.0),
            float(actor.get("mana", 0)) / 10.0,
            float(actor.get("max_mana", 0)) / 10.0,
            float(opponent.get("mana", 0)) / 10.0,
            float(opponent.get("max_mana", 0)) / 10.0,
            float(actor.get("hand_count", len(actor.get("hand") or []))) / 4.0,
            float(opponent.get("hand_count", 0)) / 4.0,
            float(actor.get("deck_count", len(actor.get("remaining_deck") or [])))
            / 9.0,
            float(opponent.get("deck_count", 0)) / 9.0,
            *board_stats(actor),
            *board_stats(opponent),
            float(actor.get("mana_draw_count_this_turn", 0)) / 2.0,
        ],
        dtype=np.float64,
    )


def metronome_features(
    state: dict[str, Any],
    *,
    action_type: str,
    legal_action_count: int,
    actor_is_p1: bool,
) -> np.ndarray:
    action = np.zeros(len(ACTION_TYPES), dtype=np.float64)
    if action_type in ACTION_TYPES:
        action[ACTION_TYPES.index(action_type)] = 1.0
    return np.concatenate(
        [
            state_scalars(state),
            action,
            np.asarray(
                [
                    math.log1p(float(legal_action_count)) / 6.0,
                    float(actor_is_p1),
                ]
            ),
        ]
    )


def metronome_features_from_trace(row: dict[str, Any]) -> np.ndarray:
    actor_player = int(row.get("actor_player", 1) or 1)
    return metronome_features(
        trace_visible_state(row["pre_state"], actor_player),
        action_type=str(row.get("action_type") or ""),
        legal_action_count=int(row.get("legal_action_count", 0) or 0),
        actor_is_p1=actor_player == 1,
    )


def cardoptimum_features(
    state: dict[str, Any],
    card_id: int,
) -> np.ndarray:
    card_id = int(card_id)
    one_hot = np.zeros(len(CARD_IDS), dtype=np.float64)
    if card_id in CARD_INDEX:
        one_hot[CARD_INDEX[card_id]] = 1.0
    actor = state.get("actor") or {}
    card = next(
        (
            item
            for item in actor.get("remaining_deck") or []
            if int(item.get("card_id", -1)) == card_id
        ),
        {},
    )
    stats = np.asarray(
        [
            float(card.get("level", 1)) / 10.0,
            float(card.get("attack", 0)) / 30.0,
            float(card.get("hp", 0)) / 30.0,
            float(card.get("mana_cost", 0)) / 10.0,
            float(card.get("skip_count", 0)) / 3.0,
        ],
        dtype=np.float64,
    )
    base = state_scalars(state)
    return np.concatenate([base, one_hot, stats, base[:7] * stats.mean()])


@dataclass(frozen=True)
class AssemblerSelection:
    deck_ids: tuple[int, ...]
    score: float
    raw_score: float
    allowed_pool_ids: tuple[int, ...]
    candidates_scored: int


class AssemblerV1:
    def __init__(self, artifact: str | Path):
        self.artifact = Path(artifact).resolve()
        self.model = _load_npz(self.artifact)

    def raw_score(
        self,
        *,
        candidate_deck_ids: Sequence[int],
        opponent_deck_ids: Sequence[int],
        allowed_pool_ids: Sequence[int],
        candidate_levels: dict[int, int] | None = None,
        opponent_levels: dict[int, int] | None = None,
    ) -> float:
        features = np.concatenate(
            [
                deck_vector(candidate_deck_ids, candidate_levels),
                deck_vector(opponent_deck_ids, opponent_levels),
                pool_vector(allowed_pool_ids),
                np.asarray(
                    [
                        len(set(candidate_deck_ids)) / 9.0,
                        len(set(opponent_deck_ids)) / 9.0,
                        1.0,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        return float(_ridge_predict(self.model, features))

    def score(
        self,
        *,
        candidate_deck_ids: Sequence[int],
        opponent_deck_ids: Sequence[int],
        allowed_pool_ids: Sequence[int],
        candidate_levels: dict[int, int] | None = None,
        opponent_levels: dict[int, int] | None = None,
    ) -> float:
        return float(
            np.clip(
                self.raw_score(
                    candidate_deck_ids=candidate_deck_ids,
                    opponent_deck_ids=opponent_deck_ids,
                    allowed_pool_ids=allowed_pool_ids,
                    candidate_levels=candidate_levels,
                    opponent_levels=opponent_levels,
                ),
                0.0,
                1.0,
            )
        )

    def select(
        self,
        *,
        candidates: Iterable[Sequence[int]],
        opponent_deck_ids: Sequence[int],
        allowed_pool_ids: Sequence[int],
        candidate_levels: dict[int, int] | None = None,
        opponent_levels: dict[int, int] | None = None,
    ) -> AssemblerSelection:
        best_deck: tuple[int, ...] | None = None
        best_score = -math.inf
        count = 0
        for candidate in candidates:
            deck = tuple(int(card_id) for card_id in candidate)
            score = self.raw_score(
                candidate_deck_ids=deck,
                opponent_deck_ids=opponent_deck_ids,
                allowed_pool_ids=allowed_pool_ids,
                candidate_levels=candidate_levels,
                opponent_levels=opponent_levels,
            )
            count += 1
            if score > best_score or (score == best_score and deck < (best_deck or deck)):
                best_deck = deck
                best_score = score
        if best_deck is None:
            raise ValueError("assembler candidates must not be empty")
        return AssemblerSelection(
            deck_ids=best_deck,
            score=float(np.clip(best_score, 0.0, 1.0)),
            raw_score=float(best_score),
            allowed_pool_ids=tuple(int(card_id) for card_id in allowed_pool_ids),
            candidates_scored=count,
        )


class CardOptimumV1:
    def __init__(self, artifact: str | Path):
        self.artifact = Path(artifact).resolve()
        self.model = _load_npz(self.artifact)

    def rank(self, state: Any, actor_id: int) -> list[dict[str, float | int]]:
        snapshot = visible_state(state, actor_id)
        deck = snapshot["actor"]["remaining_deck"]
        ranked = [
            {
                "card_id": int(card["card_id"]),
                "score": float(
                    _ridge_predict(
                        self.model,
                        cardoptimum_features(snapshot, int(card["card_id"])),
                    )
                ),
            }
            for card in deck
        ]
        return sorted(ranked, key=lambda row: (-float(row["score"]), int(row["card_id"])))

    def choose(self, state: Any, actor_id: int) -> dict[str, Any] | None:
        ranked = self.rank(state, actor_id)
        if not ranked:
            return None
        return {"selected_card_id": int(ranked[0]["card_id"]), "ranked_options": ranked}


class MetronomeV1:
    def __init__(self, artifact: str | Path):
        self.artifact = Path(artifact).resolve()
        self.model = _load_npz(self.artifact)

    def predict_ms(
        self,
        state: Any,
        actor_id: int,
        *,
        action_type: str,
        legal_action_count: int,
    ) -> dict[str, float]:
        actor_is_p1 = int(getattr(state.p1, "user_id", 1)) == int(actor_id)
        features = metronome_features(
            visible_state(state, actor_id),
            action_type=action_type,
            legal_action_count=legal_action_count,
            actor_is_p1=actor_is_p1,
        )
        return self.predict_features(features)

    def predict_trace(self, row: dict[str, Any]) -> dict[str, float]:
        return self.predict_features(metronome_features_from_trace(row))

    def predict_features(self, features: np.ndarray) -> dict[str, float]:
        predicted_log = float(_ridge_predict(self.model, features))
        residuals = self.model.get(
            "residual_log_quantiles",
            np.zeros(3, dtype=np.float32),
        )
        values = {
            "point": max(100.0, min(25_000.0, math.expm1(predicted_log))),
            "p50": max(
                100.0,
                min(25_000.0, math.expm1(predicted_log + float(residuals[0]))),
            ),
            "p90": max(
                100.0,
                min(25_000.0, math.expm1(predicted_log + float(residuals[1]))),
            ),
        }
        return values


def deck_summary(
    deck_ids: Iterable[int],
    levels: dict[str, Any] | dict[int, Any] | None,
) -> np.ndarray:
    ids = [int(card_id) for card_id in deck_ids]
    levels = levels or {}
    cards = [CARD_CATALOG[card_id] for card_id in ids if card_id in CARD_CATALOG]
    nonheroes = [card for card in cards if card.get("card_type") != "hero"]
    level_values = np.asarray(
        [float(levels.get(str(card_id), levels.get(card_id, 1))) for card_id in ids],
        dtype=np.float64,
    )
    mana = np.asarray([float(card.get("mana_cost", 0)) for card in nonheroes])
    attack = np.asarray([float(card.get("base_attack", 0)) for card in nonheroes])
    hp = np.asarray([float(card.get("base_hp", 0)) for card in nonheroes])
    hero_hp = next(
        (
            float(card.get("base_hp", 0))
            for card in cards
            if card.get("card_type") == "hero"
        ),
        0.0,
    )
    return np.asarray(
        [
            hero_hp / 50.0,
            float(np.mean(level_values)) / 10.0,
            float(np.std(level_values)) / 5.0,
            float(np.max(level_values)) / 10.0,
            float(np.mean(mana)) / 10.0,
            float(np.std(mana)) / 5.0,
            float(np.mean(attack)) / 20.0,
            float(np.mean(hp)) / 20.0,
            sum(card.get("card_type") == "potion" for card in cards) / 3.0,
        ],
        dtype=np.float64,
    )


class _TimeStampV1:
    duo = False

    def __init__(self, artifact: str | Path):
        self.artifact = Path(artifact).resolve()
        self.model = _load_npz(self.artifact)

    def predict(
        self,
        *,
        actor_deck_ids: Sequence[int],
        opponent_deck_ids: Sequence[int],
        actor_levels: dict[int, int],
        opponent_levels: dict[int, int],
        actor_starts: bool,
    ) -> dict[str, float]:
        start = float(bool(actor_starts))
        turn_parts = [
            deck_vector(actor_deck_ids, actor_levels),
            np.asarray([start]),
        ]
        if self.duo:
            turn_parts.append(deck_vector(opponent_deck_ids, opponent_levels))
        turn_features = np.concatenate(turn_parts)
        predicted_log_turns = float(
            _ridge_predict(self.model, turn_features, prefix="turn_")
        )
        duration_parts = [
            np.asarray([predicted_log_turns, start]),
            deck_summary(actor_deck_ids, actor_levels),
        ]
        if self.duo:
            duration_parts.append(
                deck_summary(opponent_deck_ids, opponent_levels)
            )
        duration_features = np.concatenate(duration_parts)
        predicted_log_duration = float(
            _ridge_predict(self.model, duration_features, prefix="duration_")
        )
        residuals = self.model.get(
            "duration_residual_log_quantiles",
            np.zeros(3, dtype=np.float32),
        )
        return {
            "turns": max(1.0, math.expm1(predicted_log_turns)),
            "duration_seconds": max(0.0, math.expm1(predicted_log_duration)),
            "duration_p50_seconds": max(
                0.0,
                math.expm1(predicted_log_duration + float(residuals[0])),
            ),
            "duration_p90_seconds": max(
                0.0,
                math.expm1(predicted_log_duration + float(residuals[1])),
            ),
        }


class TimeStampMonoV1(_TimeStampV1):
    """Predict duration from the user's deck and V5 opponent population."""

    def predict(
        self,
        *,
        actor_deck_ids: Sequence[int],
        actor_levels: dict[int, int],
        actor_starts: bool,
        opponent_deck_ids: Sequence[int] = (),
        opponent_levels: dict[int, int] | None = None,
    ) -> dict[str, float]:
        return super().predict(
            actor_deck_ids=actor_deck_ids,
            opponent_deck_ids=opponent_deck_ids,
            actor_levels=actor_levels,
            opponent_levels=opponent_levels or {},
            actor_starts=actor_starts,
        )


class TimeStampDuoV1(_TimeStampV1):
    """Predict duration from both concrete decks."""

    duo = True


class ForcedDrawRandom:
    """Delegate RNG that can force exactly one weighted draw.

    The wrapped RNG is still advanced once, keeping the random stream aligned
    between the assisted and unassisted arms of a paired benchmark.
    """

    def __init__(self, base: random.Random):
        self.base = base
        self._forced_random: float | None = None

    def __getattr__(self, name: str) -> Any:
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __deepcopy__(self, memo: dict[int, Any]) -> "ForcedDrawRandom":
        clone = type(self)(copy.deepcopy(self.base, memo))
        clone._forced_random = self._forced_random
        memo[id(self)] = clone
        return clone

    def random(self) -> float:
        natural = self.base.random()
        if self._forced_random is None:
            return natural
        forced = self._forced_random
        self._forced_random = None
        return forced

    def clear(self) -> None:
        self._forced_random = None

    def arm(self, player: Any, selected_card_id: int) -> bool:
        deck = list(getattr(player, "deck", []) or [])
        if not deck or len(getattr(player, "hand", []) or []) >= 4:
            return False
        target_index = next(
            (
                index
                for index, card in enumerate(deck)
                if int(getattr(card, "card_id", -1)) == int(selected_card_id)
            ),
            None,
        )
        if target_index is None:
            return False
        cheap_in_hand = sum(1 for card in player.hand if int(card.mana_cost) <= 2)
        expensive_in_hand = sum(1 for card in player.hand if int(card.mana_cost) >= 4)
        weights = []
        for card in deck:
            stuck = (int(getattr(card, "skip_count", 0)) + 1) * 0.5
            if int(card.mana_cost) <= 2:
                cost_bias = max(0, 1 - cheap_in_hand) * 0.3
            elif int(card.mana_cost) >= 4:
                cost_bias = max(0, 1 - expensive_in_hand) * 0.3
            else:
                cost_bias = 0.0
            weights.append(1.0 + stuck + cost_bias)
        before = sum(weights[:target_index])
        total = sum(weights)
        self._forced_random = (before + 0.5 * weights[target_index]) / total
        return True


__all__ = [
    "ACTION_TYPES",
    "AssemblerSelection",
    "AssemblerV1",
    "CARD_CATALOG",
    "CARD_IDS",
    "CardOptimumV1",
    "ForcedDrawRandom",
    "HERO_CARD_IDS",
    "MetronomeV1",
    "TimeStampDuoV1",
    "card_snapshot",
    "cardoptimum_features",
    "deck_summary",
    "deck_vector",
    "pool_vector",
    "state_scalars",
    "visible_state",
]
