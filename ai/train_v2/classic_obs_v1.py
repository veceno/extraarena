"""
State observation encoder (v1) — perspective-relative, no card identity in any slot.

Feature layout (1456 floats):

  [0:32]   Global/resource section
  [32:1312] 20 card slots × 64 (own hero, enemy hero, own board 0..6,
             enemy board 0..6, own hand 0..3)
  [1312:1456] 3 zone summaries × 48 (own deck, own graveyard, enemy graveyard)

Enemy hand is NEVER encoded in card slots or zone summaries (only hand_size in globals).
"""
from __future__ import annotations

import numpy as np

from core.state import GameState, GameStatus, MECHANICS_LIST
from ai.train_v2.classic_card_shape_v1 import CARD_SHAPE_DIM, encode_card_shape

OBS_VERSION = "classic_obs_v1"
OBS_DIM = 1456

_GLOBAL_DIM = 32
_CARD_SLOTS = 20
_ZONE_SUMMARY_DIM = 48


def encode_observation(state: GameState, player_id: int) -> np.ndarray:
    """
    Encode a perspective-relative observation.

    Returns:
        np.ndarray shape (1456,) dtype float32.
    """
    out = np.zeros(OBS_DIM, dtype=np.float32)

    me = state.p1 if state.p1.user_id == player_id else state.p2
    enemy = state.p2 if state.p1.user_id == player_id else state.p1

    _encode_globals(out, state, me, enemy)
    offset = _GLOBAL_DIM

    _encode_card_slots(out, offset, me, enemy)
    offset += _CARD_SLOTS * CARD_SHAPE_DIM

    _encode_zone_summaries(out, offset, me, enemy)

    return out.astype(np.float32)


def _encode_globals(out, state, me, enemy):

    def _norm(v, div):
        return min(v / div, 1.0)

    if state.status == GameStatus.ONGOING:
        status_idx = 0
    elif state.status == GameStatus.P1_WIN:
        status_idx = 1
    elif state.status == GameStatus.P2_WIN:
        status_idx = 2
    else:
        status_idx = 3

    out[0] = _norm(state.turn_number, 50)
    out[1] = 1.0 if state.current_turn_owner_id == me.user_id else 0.0

    out[2] = float(status_idx == 0)
    out[3] = float(status_idx == 1)
    out[4] = float(status_idx == 2)
    out[5] = float(status_idx == 3)

    out[6]  = _norm(me.mana, 10)
    out[7]  = _norm(me.max_mana, 10)
    out[8]  = _norm(len(me.hand), 4)
    out[9]  = _norm(len(me.board), 7)
    out[10] = _norm(len(me.deck), 12)
    out[11] = _norm(len(me.graveyard), 16)

    out[12] = _norm(me.hero.hp, 50)
    out[13] = _norm(me.hero.max_hp, 50)
    out[14] = _norm(sum(u.attack for u in me.board), 50)
    out[15] = _norm(sum(u.hp for u in me.board), 100)
    out[16] = _norm(sum(1 for u in me.board if u.is_ready), 7)

    out[17] = _norm(enemy.mana, 10)
    out[18] = _norm(enemy.max_mana, 10)
    out[19] = _norm(len(enemy.hand), 4)
    out[20] = _norm(len(enemy.board), 7)
    out[21] = _norm(len(enemy.deck), 12)
    out[22] = _norm(len(enemy.graveyard), 16)

    out[23] = _norm(enemy.hero.hp, 50)
    out[24] = _norm(enemy.hero.max_hp, 50)
    out[25] = _norm(sum(u.attack for u in enemy.board), 50)
    out[26] = _norm(sum(u.hp for u in enemy.board), 100)
    out[27] = _norm(sum(1 for u in enemy.board if u.is_ready), 7)

    out[28] = _norm(me.trophies, 5000)
    out[29] = _norm(enemy.trophies, 5000)

    out[30] = 0.0
    out[31] = 0.0


def _encode_card_slots(out, offset, me, enemy):

    out[offset : offset + CARD_SHAPE_DIM] = encode_card_shape(me.hero)
    offset += CARD_SHAPE_DIM

    out[offset : offset + CARD_SHAPE_DIM] = encode_card_shape(enemy.hero)
    offset += CARD_SHAPE_DIM

    for i in range(7):
        card = me.board[i] if i < len(me.board) else None
        out[offset : offset + CARD_SHAPE_DIM] = encode_card_shape(
            card, board_pos=i if card else -1
        )
        offset += CARD_SHAPE_DIM

    for i in range(7):
        card = enemy.board[i] if i < len(enemy.board) else None
        out[offset : offset + CARD_SHAPE_DIM] = encode_card_shape(
            card, board_pos=i if card else -1
        )
        offset += CARD_SHAPE_DIM

    for i in range(4):
        card = me.hand[i] if i < len(me.hand) else None
        out[offset : offset + CARD_SHAPE_DIM] = encode_card_shape(
            card, hand_pos=i if card else -1
        )
        offset += CARD_SHAPE_DIM


def _encode_zone_summaries(out, offset, me, enemy):
    _encode_one_zone(out, offset + 0 * _ZONE_SUMMARY_DIM, me.deck)
    _encode_one_zone(out, offset + 1 * _ZONE_SUMMARY_DIM, me.graveyard)
    _encode_one_zone(out, offset + 2 * _ZONE_SUMMARY_DIM, enemy.graveyard)


def _encode_one_zone(out, base, cards):
    n = len(cards)
    if n == 0:
        return

    def _norm(v, div):
        return min(v / div, 1.0)

    out[base] = _norm(n, 12)

    mana_vals = [c.mana_cost for c in cards]
    atk_vals = [c.attack for c in cards]
    hp_vals = [c.hp for c in cards]

    out[base + 1] = _norm(sum(mana_vals) / n, 10)
    out[base + 2] = _norm(sum(atk_vals) / n, 20)
    out[base + 3] = _norm(sum(hp_vals) / n, 20)

    out[base + 4] = _norm(max(mana_vals), 10)
    out[base + 5] = _norm(max(atk_vals), 20)
    out[base + 6] = _norm(max(hp_vals), 20)

    out[base + 7] = _norm(sum(atk_vals), 50)

    warriors = sum(1 for c in cards if c.card_type.value == "warrior")
    potions = sum(1 for c in cards if c.card_type.value == "potion")
    out[base + 8] = warriors / n
    out[base + 9] = potions / n

    out[base + 10] = _norm(sum(c.level for c in cards) / n, 10)

    for mi, mech_name in enumerate(MECHANICS_LIST):
        count = 0
        for c in cards:
            for cm in c.mechanics:
                if cm == mech_name or cm.startswith(mech_name + "_"):
                    count += 1
                    break
        out[base + 11 + mi] = count / n

    out[base + 44] = 0.0
    out[base + 45] = 0.0
    out[base + 46] = 0.0
    out[base + 47] = 0.0
