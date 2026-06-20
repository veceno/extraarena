"""
Card-shape encoder for RL — deliberately excludes card identity (card_id, name, UUID, rarity one-hot).

Feature layout (64 floats, stable):

  [0:3]   Type one-hot: hero, warrior, potion
  [3:14]  Numeric/status:
            mana_cost/10, attack/20, hp/20, max_hp/20, hp_fraction,
            is_ready, is_frozen, level/10,
            effective_attack/20, board_pos_norm, hand_pos_norm
  [14:47] 33 mechanic-family flags (MECHANICS_LIST order)
  [47:64] 17 parsed mechanic scalar channels:
            damage, heal, aoe_damage, battlecry_damage, battlecry_heal,
            buff_atk, buff_hp, armor, reflect, regen, aura_atk,
            cleave_dmg, cleave_targets, mana_gain, mana_drain,
            start_mana, summon_value_or_flag
"""
from __future__ import annotations

from functools import lru_cache
import re
from typing import Optional

import numpy as np

from core.state import CardInstance, CardType, MECHANICS_LIST

CARD_SHAPE_VERSION = "classic_card_shape_v1"
CARD_SHAPE_DIM = 64

_TYPE_INDEX = {CardType.HERO: 0, CardType.WARRIOR: 1, CardType.POTION: 2}

_SCALAR_PATTERNS = [
    ("damage",                     re.compile(r"(?:^|_)damage_(\d+)")),
    ("heal",                       re.compile(r"^heal(?:_target)?_(\d+)")),
    ("aoe_damage",                 re.compile(r"^aoe_damage_(\d+)")),
    ("battlecry_damage",           re.compile(r"^battlecry_damage_(\d+)")),
    ("battlecry_heal",             re.compile(r"^battlecry_heal_(?:hero_|target_)?(\d+)")),
    ("buff_atk",                   re.compile(r"^(?:battlecry_buff|buff_all|buff_atk)_(\d+)")),
    ("buff_hp",                    re.compile(r"^(?:battlecry_buff|buff_all)_\d+_(\d+)")),
    ("armor",                      re.compile(r"^armor_(\d+)")),
    ("reflect",                    re.compile(r"^reflect_(\d+)")),
    ("regen",                      re.compile(r"^regen_(\d+)")),
    ("aura_atk",                   re.compile(r"^aura_atk_(\d+)")),
    ("cleave_dmg",                 re.compile(r"^cleave_(\d+)_\d+")),
    ("cleave_targets",             re.compile(r"^cleave_\d+_(\d+)")),
    ("mana_gain",                  re.compile(r"^mana_gain_(\d+)")),
    ("mana_drain",                 re.compile(r"^mana_drain_(\d+)")),
    ("start_mana",                 re.compile(r"^start_mana_(\d+)")),
    ("summon_value_or_flag",       re.compile(r"^summon_(\d+)")),
]

SCALAR_NORMALIZERS = {
    "damage":             10.0,
    "heal":               10.0,
    "aoe_damage":         10.0,
    "battlecry_damage":   10.0,
    "battlecry_heal":     10.0,
    "buff_atk":           10.0,
    "buff_hp":            10.0,
    "armor":              10.0,
    "reflect":            10.0,
    "regen":              10.0,
    "aura_atk":           10.0,
    "cleave_dmg":         10.0,
    "cleave_targets":      5.0,
    "mana_gain":          10.0,
    "mana_drain":         10.0,
    "start_mana":         10.0,
    "summon_value_or_flag": 20.0,
}


@lru_cache(maxsize=4096)
def _encode_mechanics_cached(mechanics_tuple: tuple[str, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Cache expensive string/regex mechanics parsing for repeated card shapes."""
    flags: list[float] = []
    for mechanic_name in MECHANICS_LIST:
        has_flag = 0.0
        for card_mechanic in mechanics_tuple:
            if card_mechanic == mechanic_name or card_mechanic.startswith(mechanic_name + "_"):
                has_flag = 1.0
                break
        flags.append(has_flag)

    scalars: list[float] = []
    for scalar_name, pattern in _SCALAR_PATTERNS:
        best_val = 0.0
        for card_mechanic in mechanics_tuple:
            m = pattern.search(card_mechanic)
            if m:
                val = float(m.group(1))
                if val > best_val:
                    best_val = val
        if best_val > 0:
            norm = SCALAR_NORMALIZERS.get(scalar_name, 10.0)
            scalars.append(min(best_val / norm, 1.0))
        elif scalar_name == "summon_value_or_flag" and "summon" in mechanics_tuple:
            scalars.append(1.0)
        else:
            scalars.append(0.0)

    return tuple(flags), tuple(scalars)


def encode_card_shape(
    card: CardInstance | None,
    *,
    board_pos: int = -1,
    hand_pos: int = -1,
    effective_attack: int | None = None,
) -> np.ndarray:
    """
    Encode a card into a fixed 64-float vector with no card identity.

    Returns:
        np.ndarray shape (64,) dtype float32.
    """
    out = np.zeros(CARD_SHAPE_DIM, dtype=np.float32)

    if card is None:
        return out

    eff_atk = effective_attack if effective_attack is not None else card.attack

    type_idx = _TYPE_INDEX.get(card.card_type, 1)
    out[type_idx] = 1.0

    out[3]  = min(card.mana_cost / 10.0, 1.0)
    out[4]  = min(card.attack / 20.0, 1.0)
    out[5]  = min(card.hp / 20.0, 1.0)
    out[6]  = min(card.max_hp / 20.0, 1.0)
    out[7]  = card.hp / max(card.max_hp, 1)
    out[8]  = float(card.is_ready)
    out[9]  = float(card.is_frozen)
    out[10] = min(card.level / 10.0, 1.0)
    out[11] = min(eff_atk / 20.0, 1.0)
    out[12] = (board_pos + 1) / 8.0 if board_pos >= 0 else 0.0
    out[13] = (hand_pos + 1) / 5.0 if hand_pos >= 0 else 0.0

    flags, scalars = _encode_mechanics_cached(tuple(card.mechanics))
    out[14 : 14 + len(flags)] = flags
    out[47 : 47 + len(scalars)] = scalars

    return out
