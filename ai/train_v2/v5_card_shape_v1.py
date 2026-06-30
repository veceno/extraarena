"""
V5 card-shape encoder — Block 0 fork of the frozen classic_card_shape_v1.

This is a SEPARATE file from the byte-frozen ``classic_card_shape_v1`` (which
backs V4-orig ONNX and must never change).  V5-only mechanic families and the
index-47 overlap fix live here.

Feature layout (73 floats, grow + disjoint from classic 64):

  [0:3]    Type one-hot: hero, warrior, potion
  [3:14]   Numeric/status:
             mana_cost/10, attack/20, hp/20, max_hp/20, hp_fraction,
             is_ready, is_frozen, level/10,
             effective_attack/20, board_pos_norm, hand_pos_norm
  [14:47]  33 classic mechanic-family one-hots (MECHANICS_LIST[0:33] — the
           first 33 families; desk_freeze, MECHANICS_LIST[33], is intentionally
           NOT written here — see the index-47-overlap fix at index 69)
  [47:64]  17 classic mechanic scalar channels (damage, heal, aoe_damage, ...)
           — byte-identical to frozen classic
  [64:69]  5 NEW V5-only mechanic one-hots:
             [64] aoe_silence
             [65] team_wide_shield
             [66] rebirth            (rebirth_N)
             [67] crime_and_punishment (crime_and_punishment_N)
             [68] target_ally_max_hp_plus (target_ally_max_hp_plus[_universal]_N)
  [69]     desk_freeze one-hot — index-47-overlap FIX.  The frozen classic writes
           desk_freeze's flag at index 47 where it is ALWAYS clobbered by the
           damage scalar (classic writes flags to [14:48) then scalars to
           [47:64), so out[47] is overwritten).  V5 writes only the first 33
           flags to [14:47) and re-encodes desk_freeze here at a disjoint index,
           so both desk_freeze (index 69) and damage (index 47) survive.
  [70:73]  3 NEW V5 magnitude scalars (only the families that carry a parsed N
           parameter — grounded in core/effects.py):
             [70] rebirth_N / 10.0
             [71] crime_and_punishment_N / 10.0
             [72] target_ally_max_hp_plus_N / 10.0
           aoe_silence and team_wide_shield carry NO magnitude (hardcoded
           limit=3 / binary shield — core/effects.py:1020-1032, 1071-1086),
           so they get a one-hot only, no magnitude scalar.

Warm-start safety: [0:64) is byte-identical to the frozen classic encoder
across the entire card catalog.  The classic encoder ALWAYS overwrites index
47 with the damage scalar (the desk_freeze flag written there is invisible in
the final classic output), so not writing desk_freeze at 47 in V5 leaves
[0:64) unchanged.
"""
from __future__ import annotations

from functools import lru_cache
import re
from typing import Optional

import numpy as np

# Reuse the frozen classic internal helpers (read-only import — the frozen
# file itself is never modified).  This guarantees the [0:64) mechanics parsing
# (34 flags + 17 scalars) is byte-identical to classic.
from ai.train_v2.classic_card_shape_v1 import (
    _TYPE_INDEX,
    _encode_mechanics_cached,
)
from core.state import CardInstance, MECHANICS_LIST

CARD_SHAPE_VERSION = "v5_card_shape_v1"
CARD_SHAPE_DIM_V5 = 73

# ---------------------------------------------------------------------------
# V5-only mechanic families (NOT in the frozen 34-entry MECHANICS_LIST).
# Grounded in core/effects.py.
# ---------------------------------------------------------------------------
V5_NEW_MECHANIC_FAMILIES = (
    "aoe_silence",            # effects.py:1020  — no _N (hardcoded limit=3)
    "team_wide_shield",       # effects.py:1071  — no _N (binary shield, :1055)
    "rebirth",                # effects.py:1161  — rebirth_N (parse_rebirth)
    "crime_and_punishment",   # effects.py:1175  — crime_and_punishment_N
    "target_ally_max_hp_plus",  # effects.py:1112 — target_ally_max_hp_plus[_universal]_N
)
V5_NEW_ONEHOT_BASE = 64
V5_DESK_FREEZE_INDEX = 69
V5_MAGNITUDE_BASE = 70

# Families that carry a parsed magnitude N (grounded in effects.py).  Only
# these get a magnitude scalar; aoe_silence and team_wide_shield do not.
V5_MAGNITUDE_FAMILIES = ("rebirth", "crime_and_punishment", "target_ally_max_hp_plus")
V5_MAGNITUDE_NORMALIZER = 10.0  # consistent with classic SCALAR_NORMALIZERS

# Regex patterns grounded in core/effects.py:
#   rebirth_(\d+)$                       — effects.py:1164
#   crime_and_punishment_(\d+)$          — effects.py:1178
#   target_ally_max_hp_plus(?:_universal)?_(\d+)$  — effects.py:1143-1144
_V5_MAGNITUDE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("rebirth", re.compile(r"rebirth_(\d+)$")),
    ("crime_and_punishment", re.compile(r"crime_and_punishment_(\d+)$")),
    ("target_ally_max_hp_plus", re.compile(r"target_ally_max_hp_plus(?:_universal)?_(\d+)$")),
)


@lru_cache(maxsize=4096)
def _encode_v5_extras_cached(
    mechanics_tuple: tuple[str, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Cache V5-only one-hots, magnitude scalars, and the desk_freeze fix flag.

    Returns (new_onehots[5], magnitude_scalars[3], desk_freeze_flag).
    """
    new_onehots: list[float] = []
    for family in V5_NEW_MECHANIC_FAMILIES:
        has = 0.0
        for m in mechanics_tuple:
            if m == family or m.startswith(family + "_"):
                has = 1.0
                break
        new_onehots.append(has)

    magnitude_scalars: list[float] = []
    for _family, pattern in _V5_MAGNITUDE_PATTERNS:
        best = 0.0
        for m in mechanics_tuple:
            match = pattern.search(m)
            if match:
                val = float(match.group(1))
                if val > best:
                    best = val
        magnitude_scalars.append(min(best / V5_MAGNITUDE_NORMALIZER, 1.0))

    # desk_freeze flag — mirrors classic MECHANICS_LIST[33] matching (the flag
    # classic writes at index 47 before it is clobbered by the damage scalar).
    desk_freeze = 0.0
    for m in mechanics_tuple:
        if m == "desk_freeze" or m.startswith("desk_freeze_"):
            desk_freeze = 1.0
            break

    return tuple(new_onehots), tuple(magnitude_scalars), desk_freeze


def encode_card_shape_v5(
    card: CardInstance | None,
    *,
    board_pos: int = -1,
    hand_pos: int = -1,
    effective_attack: int | None = None,
) -> np.ndarray:
    """
    Encode a card into a fixed 73-float V5 vector with no card identity.

    [0:64) is byte-identical to the frozen ``classic_card_shape_v1`` encoder.
    [64:73) appends V5-only mechanic one-hots, the desk_freeze overlap-fix flag,
    and grounded magnitude scalars.

    Returns:
        np.ndarray shape (73,) dtype float32.
    """
    out = np.zeros(CARD_SHAPE_DIM_V5, dtype=np.float32)

    if card is None:
        return out

    eff_atk = effective_attack if effective_attack is not None else card.attack

    # --- [0:14) type + stats (re-implemented to match classic byte-for-byte) ---
    type_idx = _TYPE_INDEX.get(card.card_type, 1)
    out[type_idx] = 1.0

    out[3] = min(card.mana_cost / 10.0, 1.0)
    out[4] = min(card.attack / 20.0, 1.0)
    out[5] = min(card.hp / 20.0, 1.0)
    out[6] = min(card.max_hp / 20.0, 1.0)
    out[7] = card.hp / max(card.max_hp, 1)
    out[8] = float(card.is_ready)
    out[9] = float(card.is_frozen)
    out[10] = min(card.level / 10.0, 1.0)
    out[11] = min(eff_atk / 20.0, 1.0)
    out[12] = (board_pos + 1) / 8.0 if board_pos >= 0 else 0.0
    out[13] = (hand_pos + 1) / 5.0 if hand_pos >= 0 else 0.0

    # --- [14:47) + [47:64) classic mechanics (frozen helper => byte-identical) ---
    flags, scalars = _encode_mechanics_cached(tuple(card.mechanics))
    # Write only the first 33 flags to [14:47).  flags[33] (desk_freeze) is NOT
    # written here — it would land at index 47 and be clobbered by the damage
    # scalar, exactly as in classic.  Instead it is re-encoded at index 69.
    out[14 : 14 + 33] = flags[:33]
    # 17 scalars to [47:64) — identical to classic (the scalar block is what
    # classic writes last, so this matches classic's final [47:64) output).
    out[47 : 47 + len(scalars)] = scalars

    # --- [64:73) V5-only extensions ---
    new_onehots, magnitude_scalars, desk_freeze_flag = _encode_v5_extras_cached(
        tuple(card.mechanics)
    )
    out[V5_NEW_ONEHOT_BASE : V5_NEW_ONEHOT_BASE + 5] = new_onehots
    out[V5_DESK_FREEZE_INDEX] = desk_freeze_flag
    out[V5_MAGNITUDE_BASE : V5_MAGNITUDE_BASE + 3] = magnitude_scalars

    return out


__all__ = [
    "CARD_SHAPE_DIM_V5",
    "CARD_SHAPE_VERSION",
    "V5_DESK_FREEZE_INDEX",
    "V5_MAGNITUDE_BASE",
    "V5_MAGNITUDE_FAMILIES",
    "V5_MAGNITUDE_NORMALIZER",
    "V5_NEW_MECHANIC_FAMILIES",
    "V5_NEW_ONEHOT_BASE",
    "encode_card_shape_v5",
]