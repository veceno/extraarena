"""Production ONNX runtime for the ExtraLR V1 auxiliary models.

This module deliberately has no dependency on ``TrainV3.5``.  The feature
builders below are the serving copy of the audited Phase-C contracts:

* Assembler: candidate deck + opponent deck + allowed pool;
* CardOptimum: human-visible battle state + a candidate draw;
* Metronome: human-visible battle state + action complexity;
* TimeStamp: one- or two-deck duration estimates.

All inference goes through ONNX Runtime.  A runtime bundle owns the sessions
and can be closed explicitly during application shutdown.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "ai" / "models"
_LOG = logging.getLogger(__name__)

CARD_CATALOG = {
    int(card["id"]): card
    for card in json.loads((ROOT / "ai" / "cards.json").read_text(encoding="utf-8"))
}
CARD_IDS = tuple(sorted(CARD_CATALOG))
if len(CARD_IDS) != 50:
    raise RuntimeError(
        "ExtraLR V1 auxiliary feature contract requires the audited 50-card "
        f"catalog, found {len(CARD_IDS)} cards"
    )
CARD_INDEX = {card_id: index for index, card_id in enumerate(CARD_IDS)}
HERO_CARD_IDS = frozenset(
    card_id
    for card_id, card in CARD_CATALOG.items()
    if card.get("card_type") == "hero"
)
SIMPLIFIED_LEVEL_CARD_IDS = frozenset(
    card_id
    for card_id, card in CARD_CATALOG.items()
    if bool(card.get("simplified_levelup", False))
)
ACTION_TYPES = ("attack", "play_card", "mana_draw", "end_turn")


def _load_sidecar(path: Path, *, expected_kind: str) -> dict[str, Any]:
    sidecar_path = Path(str(path) + ".json")
    if not path.is_file():
        raise FileNotFoundError(path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "extra_lr_aux_onnx_v1":
        raise ValueError(f"{sidecar_path}: unsupported auxiliary ONNX schema")
    if payload.get("kind") != expected_kind:
        raise ValueError(
            f"{sidecar_path}: expected kind={expected_kind!r}, "
            f"got {payload.get('kind')!r}"
        )
    return payload


class _OrtModel:
    """Small validated owner for one CPU ONNX Runtime session."""

    def __init__(
        self,
        artifact: str | Path,
        *,
        kind: str,
        input_dims: dict[str, int],
        output_names: Sequence[str],
    ) -> None:
        self.artifact = Path(artifact).resolve()
        self.kind = kind
        self.sidecar = _load_sidecar(self.artifact, expected_kind=kind)
        self._output_names = tuple(output_names)
        self._session: ort.InferenceSession | None = ort.InferenceSession(
            str(self.artifact),
            providers=["CPUExecutionProvider"],
        )
        graph_inputs = {item.name: item for item in self._session.get_inputs()}
        graph_outputs = {item.name: item for item in self._session.get_outputs()}
        if set(graph_inputs) != set(input_dims):
            raise ValueError(
                f"{self.artifact.name}: inputs {sorted(graph_inputs)} do not match "
                f"{sorted(input_dims)}"
            )
        if not set(output_names).issubset(graph_outputs):
            raise ValueError(
                f"{self.artifact.name}: outputs {sorted(graph_outputs)} do not include "
                f"{sorted(output_names)}"
            )
        sidecar_inputs = self.sidecar.get("inputs") or {}
        for name, expected_dim in input_dims.items():
            graph_dim = graph_inputs[name].shape[-1]
            declared_shape = sidecar_inputs.get(name)
            declared_dim = declared_shape[-1] if declared_shape else None
            if graph_dim != expected_dim or declared_dim != expected_dim:
                raise ValueError(
                    f"{self.artifact.name}: {name} dimension mismatch "
                    f"(graph={graph_dim}, sidecar={declared_dim}, "
                    f"expected={expected_dim})"
                )

    @property
    def provenance(self) -> dict[str, str | None]:
        source_checkpoint = str(
            self.sidecar.get("source_checkpoint") or self.artifact.name
        )
        weights_hash = self.sidecar.get("source_checkpoint_sha256")
        if not weights_hash:
            digest = hashlib.sha256()
            with self.artifact.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            weights_hash = digest.hexdigest()
        return {
            "model_id": f"extra-lr-{self.kind}-v1",
            "model_family": "extra-lr-aux-v1",
            "model_version": source_checkpoint,
            "checkpoint_id": Path(source_checkpoint).stem,
            "weights_hash": str(weights_hash),
            "adapter_kind": "onnx_aux_v1",
        }

    def run(self, **inputs: np.ndarray) -> tuple[np.ndarray, ...]:
        if self._session is None:
            raise RuntimeError(f"{self.artifact.name}: ONNX session is closed")
        feeds = {
            name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in inputs.items()
        }
        outputs = tuple(
            np.asarray(value)
            for value in self._session.run(list(self._output_names), feeds)
        )
        if any(not np.isfinite(value).all() for value in outputs):
            raise RuntimeError(f"{self.artifact.name}: non-finite ONNX output")
        return outputs

    def close(self) -> None:
        # InferenceSession has no public close method.  Releasing the last
        # reference deterministically drops its native resources.
        self._session = None


def deck_vector(
    deck_ids: Iterable[int],
    levels: dict[str, Any] | dict[int, Any] | None = None,
) -> np.ndarray:
    """Encode card counts and max normalized levels (100 features)."""

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


def effective_card_level(card_id: int, raw_level: Any) -> int:
    """Normalize a collection level to the level the battle engine applies."""

    card_id = int(card_id)
    maximum = 2 if card_id in SIMPLIFIED_LEVEL_CARD_IDS else 10
    try:
        level = int(raw_level or 1)
    except (TypeError, ValueError):
        level = 1
    return max(1, min(maximum, level))


def assembler_deck_vector(
    deck_ids: Iterable[int],
    levels: dict[str, Any] | dict[int, Any] | None = None,
) -> np.ndarray:
    """Encode Assembler bags using the battle engine's effective levels."""

    out = np.zeros(len(CARD_IDS) * 2, dtype=np.float64)
    levels = levels or {}
    for raw_card_id in deck_ids:
        card_id = int(raw_card_id)
        index = CARD_INDEX.get(card_id)
        if index is None:
            continue
        out[index] += 1.0
        level = effective_card_level(
            card_id,
            levels.get(str(card_id), levels.get(card_id, 1)),
        )
        out[len(CARD_IDS) + index] = max(
            out[len(CARD_IDS) + index],
            level / 10.0,
        )
    return out


def assembler_deck_strength(
    deck_ids: Iterable[int],
    levels: dict[str, Any] | dict[int, Any] | None = None,
) -> np.ndarray:
    """Encode the card side of Assembler's 50x50 counter-card matrix."""

    out = np.zeros(len(CARD_IDS), dtype=np.float64)
    levels = levels or {}
    for raw_card_id in deck_ids:
        card_id = int(raw_card_id)
        index = CARD_INDEX.get(card_id)
        if index is None:
            continue
        maximum = 2 if card_id in SIMPLIFIED_LEVEL_CARD_IDS else 10
        level = effective_card_level(
            card_id,
            levels.get(str(card_id), levels.get(card_id, 1)),
        )
        out[index] += 0.5 + 0.5 * (level / maximum)
    return out


def pool_vector(card_ids: Iterable[int]) -> np.ndarray:
    """Encode the set of cards available to Assembler (50 features)."""

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
    """Project a live state to the human-visible auxiliary-model contract."""

    actor_id = int(actor_id)
    if int(state.p1.user_id) == actor_id:
        actor, opponent = state.p1, state.p2
    elif int(state.p2.user_id) == actor_id:
        actor, opponent = state.p2, state.p1
    else:
        raise ValueError(f"actor_id={actor_id} is not present in battle state")

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
        "actor_id": actor_id,
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
                ],
                dtype=np.float64,
            ),
        ]
    )


def cardoptimum_features(state: dict[str, Any], card_id: int) -> np.ndarray:
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
    mana = np.asarray(
        [float(card.get("mana_cost", 0)) for card in nonheroes],
        dtype=np.float64,
    )
    attack = np.asarray(
        [float(card.get("base_attack", 0)) for card in nonheroes],
        dtype=np.float64,
    )
    hp = np.asarray(
        [float(card.get("base_hp", 0)) for card in nonheroes],
        dtype=np.float64,
    )
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


def _assembler_features(
    *,
    candidate_deck_ids: Sequence[int],
    opponent_deck_ids: Sequence[int],
    allowed_pool_ids: Sequence[int],
    candidate_levels: dict[int, int] | None,
    opponent_levels: dict[int, int] | None,
) -> np.ndarray:
    candidate_strength = assembler_deck_strength(
        candidate_deck_ids,
        candidate_levels,
    )
    opponent_strength = assembler_deck_strength(
        opponent_deck_ids,
        opponent_levels,
    )
    return np.concatenate(
        [
            assembler_deck_vector(candidate_deck_ids, candidate_levels),
            assembler_deck_vector(opponent_deck_ids, opponent_levels),
            pool_vector(allowed_pool_ids),
            np.asarray(
                [
                    len(set(candidate_deck_ids)) / 9.0,
                    len(set(opponent_deck_ids)) / 9.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
            np.outer(candidate_strength, opponent_strength).reshape(-1),
        ]
    )


@dataclass(frozen=True)
class AssemblerSelection:
    deck_ids: tuple[int, ...]
    score: float
    raw_score: float
    allowed_pool_ids: tuple[int, ...]
    candidates_scored: int


class AssemblerV1(_OrtModel):
    FEATURE_DIM = 2753

    def __init__(self, artifact: str | Path) -> None:
        super().__init__(
            artifact,
            kind="assembler",
            input_dims={"features": self.FEATURE_DIM},
            output_names=("matchup_score",),
        )

    def score_candidates(
        self,
        *,
        candidates: Sequence[Sequence[int]],
        opponent_deck_ids: Sequence[int],
        allowed_pool_ids: Sequence[int],
        candidate_levels: dict[int, int] | None = None,
        candidate_slot_levels: Sequence[int] | None = None,
        opponent_levels: dict[int, int] | None = None,
    ) -> np.ndarray:
        if not candidates:
            raise ValueError("assembler candidates must not be empty")
        if candidate_levels is not None and candidate_slot_levels is not None:
            raise ValueError(
                "candidate_levels and candidate_slot_levels are mutually exclusive"
            )
        slot_levels = (
            tuple(int(level) for level in candidate_slot_levels)
            if candidate_slot_levels is not None
            else None
        )
        features = np.stack(
            [
                _assembler_features(
                    candidate_deck_ids=candidate,
                    opponent_deck_ids=opponent_deck_ids,
                    allowed_pool_ids=allowed_pool_ids,
                    candidate_levels=(
                        {
                            int(card_id): slot_levels[index]
                            for index, card_id in enumerate(candidate)
                            if index < len(slot_levels)
                        }
                        if slot_levels is not None
                        else candidate_levels
                    ),
                    opponent_levels=opponent_levels,
                )
                for candidate in candidates
            ]
        )
        return self.run(features=features)[0].reshape(-1).astype(np.float64)

    def raw_score(
        self,
        *,
        candidate_deck_ids: Sequence[int],
        opponent_deck_ids: Sequence[int],
        allowed_pool_ids: Sequence[int],
        candidate_levels: dict[int, int] | None = None,
        candidate_slot_levels: Sequence[int] | None = None,
        opponent_levels: dict[int, int] | None = None,
    ) -> float:
        return float(
            self.score_candidates(
                candidates=[candidate_deck_ids],
                opponent_deck_ids=opponent_deck_ids,
                allowed_pool_ids=allowed_pool_ids,
                candidate_levels=candidate_levels,
                candidate_slot_levels=candidate_slot_levels,
                opponent_levels=opponent_levels,
            )[0]
        )

    def score(self, **kwargs: Any) -> float:
        return float(np.clip(self.raw_score(**kwargs), 0.0, 1.0))

    def select(
        self,
        *,
        candidates: Iterable[Sequence[int]],
        opponent_deck_ids: Sequence[int],
        allowed_pool_ids: Sequence[int],
        candidate_levels: dict[int, int] | None = None,
        candidate_slot_levels: Sequence[int] | None = None,
        opponent_levels: dict[int, int] | None = None,
    ) -> AssemblerSelection:
        decks = sorted(
            {tuple(int(card_id) for card_id in candidate) for candidate in candidates}
        )
        scores = self.score_candidates(
            candidates=decks,
            opponent_deck_ids=opponent_deck_ids,
            allowed_pool_ids=allowed_pool_ids,
            candidate_levels=candidate_levels,
            candidate_slot_levels=candidate_slot_levels,
            opponent_levels=opponent_levels,
        )
        best_index = min(
            range(len(decks)),
            key=lambda index: (-float(scores[index]), decks[index]),
        )
        raw_score = float(scores[best_index])
        return AssemblerSelection(
            deck_ids=decks[best_index],
            score=float(np.clip(raw_score, 0.0, 1.0)),
            raw_score=raw_score,
            allowed_pool_ids=tuple(int(card_id) for card_id in allowed_pool_ids),
            candidates_scored=len(decks),
        )

    @staticmethod
    def generate_candidates(
        *,
        seed: int,
        allowed_card_ids: Iterable[int] | None = None,
        disabled_card_ids: Iterable[int] = (),
        candidate_count: int = 256,
    ) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
        """Generate deterministic legal 9-card decks for the current catalog."""

        disabled = {int(card_id) for card_id in disabled_card_ids}
        explicitly_allowed = (
            set(CARD_IDS)
            if allowed_card_ids is None
            else {int(card_id) for card_id in allowed_card_ids}
        )
        allowed = tuple(
            card_id
            for card_id in CARD_IDS
            if card_id in explicitly_allowed
            and card_id not in disabled
            and not bool(CARD_CATALOG[card_id].get("disabled", False))
        )
        heroes = tuple(card_id for card_id in allowed if card_id in HERO_CARD_IDS)
        nonheroes = tuple(card_id for card_id in allowed if card_id not in HERO_CARD_IDS)
        if not heroes or len(nonheroes) < 8:
            raise ValueError(
                "assembler requires at least one enabled hero and eight enabled "
                "non-hero cards"
            )
        requested = max(1, int(candidate_count))
        possible = len(heroes) * math.comb(len(nonheroes), 8)
        target = min(requested, possible)
        rng = random.Random(int(seed) ^ 0x5A55E4B1)
        decks: set[tuple[int, ...]] = set()
        while len(decks) < target:
            hero = heroes[rng.randrange(len(heroes))]
            units = tuple(sorted(rng.sample(nonheroes, 8)))
            decks.add((hero, *units))
        return allowed, sorted(decks)


class CardOptimumV1(_OrtModel):
    FEATURE_DIM = 82

    def __init__(self, artifact: str | Path) -> None:
        super().__init__(
            artifact,
            kind="cardoptimum",
            input_dims={"features": self.FEATURE_DIM},
            output_names=("card_score",),
        )

    def rank(self, state: Any, actor_id: int) -> list[dict[str, float | int]]:
        snapshot = visible_state(state, actor_id)
        deck = snapshot["actor"]["remaining_deck"]
        if not deck:
            return []
        features = np.stack(
            [
                cardoptimum_features(snapshot, int(card["card_id"]))
                for card in deck
            ]
        )
        scores = self.run(features=features)[0].reshape(-1)
        ranked = [
            {"card_id": int(card["card_id"]), "score": float(score)}
            for card, score in zip(deck, scores, strict=True)
        ]
        return sorted(
            ranked,
            key=lambda row: (-float(row["score"]), int(row["card_id"])),
        )

    def choose(self, state: Any, actor_id: int) -> dict[str, Any] | None:
        ranked = self.rank(state, actor_id)
        if not ranked:
            return None
        return {
            "selected_card_id": int(ranked[0]["card_id"]),
            "ranked_options": ranked,
        }


class MetronomeV1(_OrtModel):
    FEATURE_DIM = 26
    MIN_MS = 100.0
    MAX_MS = 25_000.0

    def __init__(self, artifact: str | Path) -> None:
        super().__init__(
            artifact,
            kind="metronome",
            input_dims={"features": self.FEATURE_DIM},
            output_names=("predicted_log_ms",),
        )
        residuals = self.sidecar.get("residual_log_quantiles") or [0.0, 0.0, 0.0]
        if len(residuals) < 3:
            raise ValueError(f"{self.artifact.name}: missing residual quantiles")
        self.residual_log_quantiles = tuple(float(value) for value in residuals[:3])

    @classmethod
    def _milliseconds(cls, predicted_log: float, residual: float = 0.0) -> float:
        return max(
            cls.MIN_MS,
            min(cls.MAX_MS, math.expm1(float(predicted_log) + float(residual))),
        )

    def _features(
        self,
        state: Any,
        actor_id: int,
        *,
        action_type: str,
        legal_action_count: int,
    ) -> np.ndarray:
        actor_is_p1 = int(state.p1.user_id) == int(actor_id)
        return metronome_features(
            visible_state(state, actor_id),
            action_type=action_type,
            legal_action_count=legal_action_count,
            actor_is_p1=actor_is_p1,
        )

    def _predict_log(
        self,
        state: Any,
        actor_id: int,
        *,
        action_type: str,
        legal_action_count: int,
    ) -> float:
        features = self._features(
            state,
            actor_id,
            action_type=action_type,
            legal_action_count=legal_action_count,
        )
        return float(self.run(features=features[None, :])[0].reshape(-1)[0])

    def predict_ms(
        self,
        state: Any,
        actor_id: int,
        *,
        action_type: str,
        legal_action_count: int,
    ) -> dict[str, float]:
        predicted_log = self._predict_log(
            state,
            actor_id,
            action_type=action_type,
            legal_action_count=legal_action_count,
        )
        q50, q90, _q99 = self.residual_log_quantiles
        return {
            "point": self._milliseconds(predicted_log),
            "p50": self._milliseconds(predicted_log, q50),
            "p90": self._milliseconds(predicted_log, q90),
        }

    def sample_ms(
        self,
        state: Any,
        actor_id: int,
        *,
        action_type: str,
        legal_action_count: int,
        rng: Any | None = None,
    ) -> float:
        """Sample a bounded human-like delay from stored residual quantiles."""

        predicted_log = self._predict_log(
            state,
            actor_id,
            action_type=action_type,
            legal_action_count=legal_action_count,
        )
        q50, q90, q99 = self.residual_log_quantiles
        # Only three empirical residual quantiles were shipped.  Mirror the
        # upper-tail distances around the median to form a conservative lower
        # tail, then use piecewise-linear inverse-CDF interpolation.
        residual = float(
            np.interp(
                float((rng or random).random()),
                (0.01, 0.10, 0.50, 0.90, 0.99),
                (2.0 * q50 - q99, 2.0 * q50 - q90, q50, q90, q99),
            )
        )
        return self._milliseconds(predicted_log, residual)


class _TimeStampV1(_OrtModel):
    duo = False
    TURN_FEATURE_DIM = 101
    DURATION_CONTEXT_DIM = 10
    KIND = "timestamp_mono"

    def __init__(self, artifact: str | Path) -> None:
        super().__init__(
            artifact,
            kind=self.KIND,
            input_dims={
                "turn_features": self.TURN_FEATURE_DIM,
                "duration_context": self.DURATION_CONTEXT_DIM,
            },
            output_names=("predicted_log_turns", "predicted_log_duration"),
        )
        residuals = self.sidecar.get("duration_residual_log_quantiles") or [
            0.0,
            0.0,
            0.0,
        ]
        if len(residuals) < 3:
            raise ValueError(f"{self.artifact.name}: missing duration quantiles")
        self.duration_residual_log_quantiles = tuple(
            float(value) for value in residuals[:3]
        )

    def predict(
        self,
        *,
        actor_deck_ids: Sequence[int],
        opponent_deck_ids: Sequence[int] = (),
        actor_levels: dict[int, int] | None = None,
        opponent_levels: dict[int, int] | None = None,
        actor_starts: bool,
    ) -> dict[str, float]:
        actor_levels = actor_levels or {}
        opponent_levels = opponent_levels or {}
        start = float(bool(actor_starts))
        turn_parts = [
            deck_vector(actor_deck_ids, actor_levels),
            np.asarray([start], dtype=np.float64),
        ]
        if self.duo:
            turn_parts.append(deck_vector(opponent_deck_ids, opponent_levels))
        duration_parts = [
            np.asarray([start], dtype=np.float64),
            deck_summary(actor_deck_ids, actor_levels),
        ]
        if self.duo:
            duration_parts.append(
                deck_summary(opponent_deck_ids, opponent_levels)
            )
        predicted_turns, predicted_duration = self.run(
            turn_features=np.concatenate(turn_parts)[None, :],
            duration_context=np.concatenate(duration_parts)[None, :],
        )
        predicted_log_turns = float(predicted_turns.reshape(-1)[0])
        predicted_log_duration = float(predicted_duration.reshape(-1)[0])
        q50, q90, _q99 = self.duration_residual_log_quantiles
        return {
            "turns": max(1.0, math.expm1(predicted_log_turns)),
            "duration_seconds": max(0.0, math.expm1(predicted_log_duration)),
            "duration_p50_seconds": max(
                0.0,
                math.expm1(predicted_log_duration + q50),
            ),
            "duration_p90_seconds": max(
                0.0,
                math.expm1(predicted_log_duration + q90),
            ),
        }


class TimeStampMonoV1(_TimeStampV1):
    """Predict a duration from the player's deck and the V5 population."""

    def predict(
        self,
        *,
        actor_deck_ids: Sequence[int],
        actor_levels: dict[int, int] | None = None,
        actor_starts: bool,
        opponent_deck_ids: Sequence[int] = (),
        opponent_levels: dict[int, int] | None = None,
    ) -> dict[str, float]:
        return super().predict(
            actor_deck_ids=actor_deck_ids,
            opponent_deck_ids=opponent_deck_ids,
            actor_levels=actor_levels,
            opponent_levels=opponent_levels,
            actor_starts=actor_starts,
        )


class TimeStampDuoV1(_TimeStampV1):
    """Predict a duration from both concrete decks."""

    duo = True
    TURN_FEATURE_DIM = 201
    DURATION_CONTEXT_DIM = 19
    KIND = "timestamp_duo"


class CardOptimumDrawRng:
    """Per-match RNG wrapper that can force one CardOptimum-selected draw.

    ``random()`` always consumes exactly one value from the wrapped generator,
    even when a forced draw is armed.  Any inference/state error fails open to
    the natural value and never leaks pending state into another match.
    """

    def __init__(
        self,
        base_rng: random.Random,
        *,
        cardoptimum: CardOptimumV1,
        state: Any,
        assisted_player_id: int,
    ) -> None:
        self.base = base_rng
        self._cardoptimum = cardoptimum
        self._state = state
        self.assisted_player_id = int(assisted_player_id)
        self._forced_random: float | None = None
        self.last_decision: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __deepcopy__(self, memo: dict[int, Any]) -> "CardOptimumDrawRng":
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        clone.base = copy.deepcopy(self.base, memo)
        clone._cardoptimum = self._cardoptimum
        clone._state = copy.deepcopy(self._state, memo)
        clone.assisted_player_id = self.assisted_player_id
        clone._forced_random = self._forced_random
        clone.last_decision = copy.deepcopy(self.last_decision, memo)
        return clone

    def bind_state(self, state: Any) -> None:
        """Rebind after an ArenaEnvironment rollback/reset swaps GameState."""

        self._state = state
        self.clear()

    def random(self) -> float:
        natural = float(self.base.random())
        forced = self._forced_random
        self._forced_random = None
        if forced is None or not math.isfinite(forced):
            return natural
        return max(0.0, min(math.nextafter(1.0, 0.0), float(forced)))

    def clear(self) -> None:
        self._forced_random = None
        self.last_decision = None

    def _arm(self, player: Any, selected_card_id: int) -> bool:
        deck = list(getattr(player, "deck", []) or [])
        hand = list(getattr(player, "hand", []) or [])
        if not deck or len(hand) >= 4:
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
        cheap_in_hand = sum(1 for card in hand if int(card.mana_cost) <= 2)
        expensive_in_hand = sum(1 for card in hand if int(card.mana_cost) >= 4)
        weights: list[float] = []
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

    def prepare_draw(self, player: Any) -> bool:
        """Arm the next weighted draw for the assisted player, or fail open."""

        self.clear()
        try:
            player_id = int(getattr(player, "user_id"))
            if player_id != self.assisted_player_id:
                return False
            choice = self._cardoptimum.choose(self._state, player_id)
            if not choice:
                return False
            selected = int(choice["selected_card_id"])
            armed = self._arm(player, selected)
            self.last_decision = {
                "armed": bool(armed),
                "selected_card_id": selected,
                "top_score": float(choice["ranked_options"][0]["score"]),
            }
            return armed
        except Exception as exc:  # CardOptimum is beta: gameplay must continue.
            _LOG.warning("CardOptimum draw assist failed open: %s", exc)
            self.clear()
            return False


@dataclass
class ExtraLRAuxRuntime:
    assembler: AssemblerV1 | None
    cardoptimum: CardOptimumV1 | None
    metronome: MetronomeV1 | None
    timestamp_mono: TimeStampMonoV1 | None
    timestamp_duo: TimeStampDuoV1 | None

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
    ) -> "ExtraLRAuxRuntime":
        directory = Path(model_dir).resolve()
        def load(model_type: type[_OrtModel], filename: str) -> _OrtModel | None:
            try:
                return model_type(directory / filename)
            except Exception as exc:
                # Component isolation is load-bearing: experimental TimeStamp
                # must not disable Metronome, and a beta CardOptimum failure
                # must not disable the base V4/V5 policy sessions.
                _LOG.error("ExtraLR auxiliary component unavailable (%s): %s", filename, exc)
                return None

        return cls(
            assembler=load(
                AssemblerV1,
                "extra_lr_assembler_v1.onnx",
            ),
            cardoptimum=load(
                CardOptimumV1,
                "extra_lr_cardoptimum_v1.onnx",
            ),
            metronome=load(
                MetronomeV1,
                "extra_lr_metronome_v1.onnx",
            ),
            timestamp_mono=load(
                TimeStampMonoV1,
                "extra_lr_timestamp_v1_mono.onnx",
            ),
            timestamp_duo=load(
                TimeStampDuoV1,
                "extra_lr_timestamp_v1_duo.onnx",
            ),
        )

    @property
    def availability(self) -> dict[str, bool]:
        return {
            "assembler": self.assembler is not None,
            "cardoptimum": self.cardoptimum is not None,
            "metronome": self.metronome is not None,
            "timestamp_mono": self.timestamp_mono is not None,
            "timestamp_duo": self.timestamp_duo is not None,
        }

    def dataset_provenance(
        self,
        *,
        include_policy_assists: bool,
        include_metronome: bool = True,
    ) -> dict[str, dict[str, str | None]]:
        """Describe only auxiliary models that affected this battle."""

        components: dict[str, _OrtModel | None] = {}
        if include_policy_assists:
            components.update(
                {
                    "assembler": self.assembler,
                    "cardoptimum": self.cardoptimum,
                }
            )
        if include_metronome:
            components["metronome"] = self.metronome
        return {
            name: model.provenance
            for name, model in components.items()
            if model is not None
        }

    def assemble_deck(
        self,
        opponent_deck_ids: Sequence[int],
        *,
        seed: int,
        allowed_card_ids: Iterable[int] | None = None,
        disabled_card_ids: Iterable[int] = (),
        candidate_count: int = 256,
        candidate_slot_levels: Sequence[int] | None = None,
        opponent_levels: dict[int, int] | None = None,
    ) -> dict[str, Any]:
        if self.assembler is None:
            raise RuntimeError("ExtraLR Assembler V1 is unavailable")
        opponent_ids = tuple(int(card_id) for card_id in opponent_deck_ids)
        allowed_pool, candidates = self.assembler.generate_candidates(
            # Keep the candidate frontier fixed for a match seed. Opponent
            # conditioning belongs in the learned 50x50 interaction matrix,
            # not in an order-sensitive candidate-generation workaround.
            seed=int(seed),
            allowed_card_ids=allowed_card_ids,
            disabled_card_ids=disabled_card_ids,
            candidate_count=candidate_count,
        )
        selection = self.assembler.select(
            candidates=candidates,
            opponent_deck_ids=opponent_ids,
            allowed_pool_ids=allowed_pool,
            candidate_slot_levels=candidate_slot_levels,
            opponent_levels=opponent_levels,
        )
        return {
            "deck_ids": list(selection.deck_ids),
            "telemetry": {
                "model": "ExtraLR Assembler V1",
                "seed": int(seed),
                "feature_schema": "assembler_bilinear_counter_v1",
                "score": selection.score,
                "raw_score": selection.raw_score,
                "allowed_pool_ids": list(selection.allowed_pool_ids),
                "candidates_scored": selection.candidates_scored,
            },
        }

    def wrap_draw_rng(
        self,
        base_rng: random.Random,
        *,
        state: Any,
        assisted_player_id: int,
    ) -> CardOptimumDrawRng:
        if self.cardoptimum is None:
            raise RuntimeError("ExtraLR CardOptimum V1 is unavailable")
        return CardOptimumDrawRng(
            base_rng,
            cardoptimum=self.cardoptimum,
            state=state,
            assisted_player_id=assisted_player_id,
        )

    def close(self) -> None:
        for model in (
            self.assembler,
            self.cardoptimum,
            self.metronome,
            self.timestamp_mono,
            self.timestamp_duo,
        ):
            if model is not None:
                model.close()


__all__ = [
    "ACTION_TYPES",
    "AssemblerSelection",
    "AssemblerV1",
    "CARD_CATALOG",
    "CARD_IDS",
    "CARD_INDEX",
    "CardOptimumDrawRng",
    "CardOptimumV1",
    "DEFAULT_MODEL_DIR",
    "ExtraLRAuxRuntime",
    "HERO_CARD_IDS",
    "MetronomeV1",
    "TimeStampDuoV1",
    "TimeStampMonoV1",
    "SIMPLIFIED_LEVEL_CARD_IDS",
    "assembler_deck_strength",
    "assembler_deck_vector",
    "card_snapshot",
    "cardoptimum_features",
    "deck_summary",
    "deck_vector",
    "effective_card_level",
    "metronome_features",
    "pool_vector",
    "state_scalars",
    "visible_state",
]
