"""V5 observation encoder for TrainV3 environment experiments."""
from __future__ import annotations

from typing import Any

import numpy as np

from ai.train_v2.classic_actions_v1 import _get_me_enemy
from ai.train_v2.classic_obs_v1 import encode_observation
from ai.train_v2.v5_card_shape_v1 import CARD_SHAPE_DIM_V5, encode_card_shape_v5
from core.state import CardInstance, CardType

from .contracts import (
    AssistModeV5,
    ENEMY_DECK_SLOTS,
    ENEMY_HAND_SLOTS,
    HISTORY_EVENT_DIM,
    HISTORY_EVENT_SOURCE_OFFSET,
    HISTORY_EVENTS,
    InfoModeV5,
    OBS_V1_DIM,
    OBS_V5_DIM,
    OWN_DECK_SLOTS,
    OWN_HAND_SLOTS,
    PRIVATE_CARD_SLOT_DIM,
    PRIVATE_INFO_DIM,
    V5_GLOBAL_DIM,
)

CARD_ID_NORMALIZER = 1000.0

# Normalizer for the mana_draw_count_this_turn global channel (spec §6.184).
# Grounded in the ruleset: max_mana is capped at 10 (core/engine.py:695) and
# each player-initiated mana draw costs MANA_DRAW_BASE * (count + 1) = 2*(count+1)
# (core/engine.py:784), so the mana pool holds max_mana / MANA_DRAW_BASE = 10/2
# = 5 base-cost units. The per-turn draw count is therefore normalized by 5 —
# count >= 5 clips to 1.0, covering the realistic 0..3 per-turn range with
# resolution. Mirrors the classic globals convention `_norm(v, div) =
# min(v/div, 1.0)` (classic_obs_v1._encode_globals) and the Rust kernel
# `norm(value, divisor)` (kernel.rs) so Python<->Rust stay byte-for-byte.
MANA_DRAW_COUNT_NORMALIZER = 5.0


def encode_observation_v5(
    state,
    player_id: int,
    *,
    info_mode: InfoModeV5 | None = None,
    assist_mode: AssistModeV5 | None = None,
    history_events: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    info_mode = info_mode or InfoModeV5()
    assist_mode = assist_mode or AssistModeV5()
    out = np.zeros(OBS_V5_DIM, dtype=np.float32)
    out[:OBS_V1_DIM] = encode_observation(state, player_id)

    me, _ = _get_me_enemy(state, player_id)
    gbase = OBS_V1_DIM
    _encode_globals_v5(
        out[gbase : gbase + V5_GLOBAL_DIM], info_mode, assist_mode, history_events, me, state, player_id
    )

    pbase = gbase + V5_GLOBAL_DIM
    _encode_private_info(out[pbase : pbase + PRIVATE_INFO_DIM], state, player_id, info_mode)

    hbase = pbase + PRIVATE_INFO_DIM
    _encode_history(out[hbase:], player_id, history_events or [])
    return out


def _encode_globals_v5(
    dst: np.ndarray,
    info_mode: InfoModeV5,
    assist_mode: AssistModeV5,
    history_events,
    me,
    state,
    player_id: int,
) -> None:
    dst[0] = info_mode.clipped_strength()
    dst[1] = float(info_mode.own_hand_identity_known)
    dst[2] = float(info_mode.own_deck_known)
    dst[3] = float(info_mode.enemy_hand_known)
    dst[4] = float(info_mode.enemy_deck_known)
    dst[5] = float(info_mode.enemy_deck_order_known)
    dst[6] = min(len(history_events or []) / HISTORY_EVENTS, 1.0)
    dst[7] = float(info_mode.draw_assist_enabled)
    dst[8] = info_mode.clipped_draw_assist_strength()
    dst[9] = float(assist_mode.assembler_enabled)
    dst[10] = assist_mode.clipped_assembler_strength()
    dst[11] = float(assist_mode.desirerer_enabled)
    dst[12] = assist_mode.clipped_desirerer_strength()
    dst[13] = float(assist_mode.teacher_hint_available)
    dst[14] = assist_mode.clipped_profile_id() / 16.0
    # mana_draw_count_this_turn channel (spec §6.184): the count of player-
    # initiated mana draws taken this turn (core/state.py:144), reset to 0 at
    # the start of each owner turn (core/engine.py:699). Normalized by
    # MANA_DRAW_COUNT_NORMALIZER (5.0) and clipped to [0, 1]; matches the Rust
    # kernel `norm(mana_draw_count_this_turn, MANA_DRAW_COUNT_NORMALIZER)`.
    dst[15] = min(float(me.mana_draw_count_this_turn) / MANA_DRAW_COUNT_NORMALIZER, 1.0)
    # Persistent turn-order channel: whether this policy side started the game.
    # classic_obs_v1 already exposes "it is my turn now"; Block B needs the
    # separate first/second-start bit so a feedforward policy can condition on
    # the structural first-move tempo across the whole episode.
    dst[16] = float(_starting_player_id(state) == int(player_id))


def _starting_player_id(state) -> int | None:
    explicit = getattr(state, "starting_player_id", None)
    if explicit in (getattr(state.p1, "user_id", 1), getattr(state.p2, "user_id", 2)):
        return int(explicit)
    if int(getattr(state, "turn_number", 0) or 0) == 1:
        p1_mana = int(getattr(state.p1, "max_mana", 0) or 0)
        p2_mana = int(getattr(state.p2, "max_mana", 0) or 0)
        if p1_mana != p2_mana:
            return int(state.p1.user_id if p1_mana > p2_mana else state.p2.user_id)
        return int(getattr(state, "current_turn_owner_id", 0) or 0)
    current = int(getattr(state, "current_turn_owner_id", 0) or 0)
    if int(getattr(state, "turn_number", 0) or 0) % 2 == 1:
        return current
    if current == int(state.p1.user_id):
        return int(state.p2.user_id)
    return int(state.p1.user_id)


def _encode_private_info(dst: np.ndarray, state, player_id: int, info_mode: InfoModeV5) -> None:
    me, enemy = _get_me_enemy(state, player_id)
    offset = 0
    offset = _encode_zone(dst, offset, me.hand, OWN_HAND_SLOTS, known=info_mode.own_hand_identity_known)
    offset = _encode_zone(dst, offset, me.deck, OWN_DECK_SLOTS, known=info_mode.own_deck_known)
    offset = _encode_zone(dst, offset, enemy.hand, ENEMY_HAND_SLOTS, known=info_mode.enemy_hand_known)
    _encode_zone(dst, offset, enemy.deck, ENEMY_DECK_SLOTS, known=info_mode.enemy_deck_known)


def _encode_zone(dst: np.ndarray, offset: int, cards, slots: int, *, known: bool) -> int:
    for slot in range(slots):
        base = offset + slot * PRIVATE_CARD_SLOT_DIM
        if known and slot < len(cards):
            card = cards[slot]
            dst[base] = 1.0
            dst[base + 1] = min(max(float(card.card_id), 0.0) / CARD_ID_NORMALIZER, 1.0)
            dst[base + 2 : base + 2 + CARD_SHAPE_DIM_V5] = encode_card_shape_v5(card)
    return offset + slots * PRIVATE_CARD_SLOT_DIM


def _encode_history(dst: np.ndarray, player_id: int, events: list[dict[str, Any]]) -> None:
    recent = events[-HISTORY_EVENTS:]
    start = HISTORY_EVENTS - len(recent)
    for idx, event in enumerate(recent):
        base = (start + idx) * HISTORY_EVENT_DIM
        _encode_one_event(dst[base : base + HISTORY_EVENT_DIM], player_id, event)


def _encode_one_event(dst: np.ndarray, player_id: int, event: dict[str, Any]) -> None:
    actor_id = int(event.get("actor_id", 0) or 0)
    # ``type`` fallback keeps pre-v5_history_event_v1 in-memory matches
    # readable while newly accepted transitions provide the full rich event.
    action_type = str(event.get("action_type") or event.get("type") or "")
    dst[0] = 1.0
    dst[1] = float(actor_id == player_id)
    dst[2] = float(actor_id not in (0, player_id))
    dst[3] = float(action_type == "end_turn")
    dst[4] = float(action_type == "play_card")
    dst[5] = float(action_type == "attack")
    dst[6] = min(max(float(event.get("action_id", 0) or 0), 0.0) / 600.0, 1.0)
    dst[7] = _signed_norm(event.get("enemy_hero_hp_delta", 0.0), 50.0)
    dst[8] = _signed_norm(event.get("own_hero_hp_delta", 0.0), 50.0)
    dst[9] = _signed_norm(event.get("my_board_count_delta", 0.0), 7.0)
    dst[10] = _signed_norm(event.get("enemy_board_count_delta", 0.0), 7.0)
    dst[11] = min(max(float(event.get("turn_number", 0) or 0), 0.0) / 50.0, 1.0)
    dst[12] = _signed_norm(event.get("board_power_delta", 0.0), 200.0)
    # Mana draw lives outside the frozen 601-action candidate codec.  Rust
    # Block-B and Phase-C Python replay reserve metadata slot 13 so it cannot
    # collapse into an arbitrary action_id=0 event.
    dst[13] = float(action_type == "mana_draw")

    source_card = _coerce_history_card(event.get("source_card"))
    target_card = _coerce_history_card(event.get("target_card"))
    src_off = HISTORY_EVENT_SOURCE_OFFSET
    tgt_off = HISTORY_EVENT_SOURCE_OFFSET + CARD_SHAPE_DIM_V5
    if source_card is not None:
        dst[src_off : src_off + CARD_SHAPE_DIM_V5] = encode_card_shape_v5(source_card)
    if target_card is not None:
        dst[tgt_off : tgt_off + CARD_SHAPE_DIM_V5] = encode_card_shape_v5(target_card)


def _coerce_history_card(card: Any) -> CardInstance | None:
    """Accept both live CardInstance objects and JSON-safe trace snapshots."""

    if card is None or isinstance(card, CardInstance):
        return card
    if not isinstance(card, dict):
        return None

    raw_type = card.get("card_type", card.get("type", CardType.WARRIOR.value))
    if isinstance(raw_type, CardType):
        card_type = raw_type
    else:
        normalized_type = str(raw_type or CardType.WARRIOR.value).lower()
        if normalized_type.startswith("cardtype."):
            normalized_type = normalized_type.split(".", 1)[1]
        try:
            card_type = CardType(normalized_type)
        except ValueError:
            card_type = CardType.WARRIOR

    return CardInstance(
        card_id=int(card.get("card_id", 0) or 0),
        card_type=card_type,
        mana_cost=int(card.get("mana_cost", 0) or 0),
        attack=int(card.get("attack", 0) or 0),
        hp=int(card.get("hp", 0) or 0),
        max_hp=int(card.get("max_hp", 0) or 0),
        mechanics=[str(value) for value in (card.get("mechanics") or [])],
        is_ready=bool(card.get("is_ready", False)),
        is_frozen=bool(card.get("is_frozen", False)),
        level=int(card.get("level", 1) or 1),
    )


def _signed_norm(value: Any, divisor: float) -> float:
    raw = float(value or 0.0) / divisor
    return max(-1.0, min(1.0, raw))


__all__ = [
    "CARD_ID_NORMALIZER",
    "MANA_DRAW_COUNT_NORMALIZER",
    "encode_observation_v5",
]
