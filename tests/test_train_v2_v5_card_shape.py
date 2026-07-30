"""
Tests for the V5 73-dim card-shape encoder (Block 0 fork of frozen classic).

Source-vs-source oracle: the Python ``encode_card_shape_v5`` is the source of
truth for the golden fixtures.  Rust mirrors the same layout byte-for-byte
(see ``card_shape_v5.rs`` unit tests).  These tests assert the Python encoder
matches the spec at the same indices the Rust tests assert, establishing
Rust<->Python parity transitively (Rust matches the fixtures, which come from
Python).

Constraints validated:
  - FROZEN-CLASSIC GUARD: ``classic_card_shape_v1`` is never modified; V5 work
    lives in the separate ``v5_card_shape_v1``.
  - [0:64) is byte-identical to the frozen classic across the whole catalog.
  - CARD_SHAPE_DIM_V5 = 73 (grow + disjoint).
  - Index-47 overlap fix: desk_freeze re-encoded at 69, damage scalar at 47.
  - 5 new mechanic one-hots at [64:69), 3 magnitude scalars at [70:73).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from core.state import CardInstance, CardType

from ai.train_v2.classic_card_shape_v1 import CARD_SHAPE_DIM, encode_card_shape
from ai.train_v2.v5_card_shape_v1 import (
    CARD_SHAPE_DIM_V5,
    V5_DESK_FREEZE_INDEX,
    V5_MAGNITUDE_BASE,
    V5_NEW_ONEHOT_BASE,
    encode_card_shape_v5,
)


# ============================================================================
# DIMENSION
# ============================================================================

class TestV5CardShapeDimension:
    def test_v5_card_shape_dim_is_73(self):
        assert CARD_SHAPE_DIM_V5 == 73

    def test_classic_dim_stays_frozen_64(self):
        # Frozen classic stays 64 — V5 grows + disjoint, never mutates classic.
        assert CARD_SHAPE_DIM == 64

    def test_v5_dim_is_classic_plus_9(self):
        # 64 classic + 5 new one-hots + 1 desk_freeze fix + 3 magnitude = 73.
        assert CARD_SHAPE_DIM_V5 == CARD_SHAPE_DIM + 9


# ============================================================================
# NONE / EMPTY
# ============================================================================

class TestV5NoneCard:
    def test_v5_encoder_handles_none_card(self):
        out = encode_card_shape_v5(None)
        assert out.shape == (CARD_SHAPE_DIM_V5,)
        assert out.dtype == np.float32
        assert np.array_equal(out, np.zeros(CARD_SHAPE_DIM_V5, dtype=np.float32))


# ============================================================================
# [0:64) BYTE-IDENTICAL TO FROZEN CLASSIC
# ============================================================================

def _load_catalog() -> dict[int, dict]:
    """Load cards.json, parsing mechanics strings into lists (mirrors ClassicRLEnv)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (
        os.path.join(here, "..", "cards.json"),
        os.path.join(here, "cards.json"),
    ):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            catalog: dict[int, dict] = {}
            for item in raw:
                cid = item.get("id", 0)
                if cid <= 0:
                    continue
                item = dict(item)
                mech = item.get("mechanics", [])
                if isinstance(mech, str):
                    try:
                        mech = json.loads(mech)
                    except (json.JSONDecodeError, TypeError):
                        mech = []
                item["mechanics"] = mech
                catalog[cid] = item
            return catalog
    return {}


def _card_from_catalog_data(data: dict) -> CardInstance:
    return CardInstance(
        card_id=data.get("id", 0),
        name=data.get("name", ""),
        card_type=CardType(data.get("card_type", "warrior")),
        mana_cost=int(data.get("mana_cost", data.get("base_mana_cost", 0)) or 0),
        attack=int(data.get("attack", data.get("base_attack", 0)) or 0),
        hp=int(data.get("hp", data.get("base_hp", 0)) or 0),
        max_hp=int(data.get("max_hp", data.get("base_max_hp", 0)) or 0),
        mechanics=list(data.get("mechanics", [])),
        level=int(data.get("level", 1) or 1),
    )


class TestV5ByteIdenticalToClassic:
    def test_v5_card_shape_matches_classic_on_0_64_across_catalog(self):
        """[0:64) must be byte-identical to the frozen classic encoder across the
        entire card catalog — the warm-start safety guarantee."""
        catalog = _load_catalog()
        assert len(catalog) >= 30, "catalog should have ~50 cards"
        for cid, data in sorted(catalog.items()):
            card = _card_from_catalog_data(data)
            v5 = encode_card_shape_v5(card, board_pos=0)
            classic = encode_card_shape(card, board_pos=0)
            assert v5.shape == (CARD_SHAPE_DIM_V5,)
            assert classic.shape == (CARD_SHAPE_DIM,)
            np.testing.assert_array_equal(
                v5[:CARD_SHAPE_DIM],
                classic,
                err_msg=f"v5/classic [0:64) divergence for card_id={cid} "
                f"({data.get('name', '?')}) mechanics={card.mechanics}",
            )

    def test_v5_card_shape_matches_classic_on_0_64_with_hand_and_eff_attack(self):
        """Byte-identical with hand_pos and effective_attack set too."""
        catalog = _load_catalog()
        for cid, data in sorted(catalog.items()):
            card = _card_from_catalog_data(data)
            v5 = encode_card_shape_v5(card, hand_pos=2, effective_attack=7)
            classic = encode_card_shape(card, hand_pos=2, effective_attack=7)
            np.testing.assert_array_equal(
                v5[:CARD_SHAPE_DIM], classic,
                err_msg=f"[0:64) divergence for card_id={cid} with hand_pos/eff_atk",
            )


# ============================================================================
# INDEX-47 OVERLAP FIX
# ============================================================================

class TestV5OverlapFix:
    def test_overlap_fixed_no_index_collision(self):
        """A card carrying desk_freeze + damage_5 must encode BOTH at disjoint
        indices — the damage scalar at 47 (classic) and the desk_freeze one-hot
        at 69 (V5 fix).  Neither clobbers the other."""
        card = CardInstance(
            card_type=CardType.POTION,
            mana_cost=1,
            attack=0,
            hp=0,
            max_hp=0,
            mechanics=["desk_freeze", "damage_5"],
        )
        out = encode_card_shape_v5(card)
        # damage_5 scalar at classic index 47 = 5/10 = 0.5 (NOT clobbered).
        assert out[47] == pytest.approx(0.5), "damage scalar at index 47"
        # desk_freeze one-hot at V5 index 69 = 1.0 (NOT clobbered by damage).
        assert out[V5_DESK_FREEZE_INDEX] == pytest.approx(1.0), \
            "desk_freeze one-hot at disjoint index 69"
        # Both features survive at disjoint indices.
        assert out[47] != 0.0
        assert out[V5_DESK_FREEZE_INDEX] != 0.0

    def test_desk_freeze_alone_is_fixed_not_lost(self):
        """Classic clobbers desk_freeze at 47 to 0.0; V5 re-encodes at 69."""
        card = CardInstance(
            card_type=CardType.POTION,
            mana_cost=1,
            attack=0,
            hp=0,
            max_hp=0,
            mechanics=["desk_freeze"],
        )
        out = encode_card_shape_v5(card)
        assert out[47] == pytest.approx(0.0), \
            "classic index 47 = 0.0 (clobbered, as in classic)"
        assert out[V5_DESK_FREEZE_INDEX] == pytest.approx(1.0), \
            "V5 index 69 = 1.0 (desk_freeze fixed)"

    def test_classic_still_clobbers_desk_freeze_at_47(self):
        """The frozen classic encoder still writes desk_freeze at 47 then clobbers
        it with the damage scalar — proving [0:64) stays byte-identical."""
        card = CardInstance(
            card_type=CardType.POTION,
            mana_cost=1,
            attack=0,
            hp=0,
            max_hp=0,
            mechanics=["desk_freeze", "damage_5"],
        )
        classic = encode_card_shape(card)
        v5 = encode_card_shape_v5(card)
        # Classic index 47 = damage scalar (0.5), desk_freeze invisible.
        assert classic[47] == pytest.approx(0.5)
        # V5 [0:64) byte-identical => index 47 also 0.5.
        assert v5[47] == pytest.approx(0.5)
        np.testing.assert_array_equal(v5[:CARD_SHAPE_DIM], classic)


# ============================================================================
# 5 NEW MECHANIC ONE-HOTS
# ============================================================================

class TestV5NewMechanicOneHots:
    @pytest.mark.parametrize(
        "mech,expected_idx",
        [
            ("aoe_silence", 64),
            ("team_wide_shield", 65),
            ("rebirth_1", 66),
            ("crime_and_punishment_2", 67),
            ("target_ally_max_hp_plus_universal_1", 68),
        ],
    )
    def test_five_new_mechanic_one_hots(self, mech, expected_idx):
        """Each new family sets its flag at [64:69) and leaves the others 0.0."""
        card = CardInstance(
            card_type=CardType.WARRIOR,
            mana_cost=1,
            attack=1,
            hp=1,
            max_hp=1,
            mechanics=[mech],
        )
        out = encode_card_shape_v5(card)
        assert out[expected_idx] == pytest.approx(1.0), \
            f"family flag at {expected_idx} for {mech}"
        # The other 4 new one-hots must be 0.0.
        for i in range(V5_NEW_ONEHOT_BASE, V5_NEW_ONEHOT_BASE + 5):
            if i != expected_idx:
                assert out[i] == pytest.approx(0.0), \
                    f"other new one-hot at {i} must be 0.0 for {mech}"

    def test_new_onehot_base_is_64(self):
        assert V5_NEW_ONEHOT_BASE == 64

    def test_desk_freeze_index_is_69(self):
        assert V5_DESK_FREEZE_INDEX == 69


# ============================================================================
# MAGNITUDE SCALARS (grounded in core/effects.py)
# ============================================================================

class TestV5MagnitudeScalars:
    def test_rebirth_magnitude(self):
        """rebirth_N -> N/10 at index 70. Grounded: effects.py:1164
        parse_rebirth uses re.match(r'rebirth_(\\d+)$', m)."""
        card = CardInstance(
            card_type=CardType.WARRIOR,
            mana_cost=8,
            attack=3,
            hp=7,
            max_hp=7,
            mechanics=["rebirth_1"],
        )
        out = encode_card_shape_v5(card)
        assert out[V5_MAGNITUDE_BASE] == pytest.approx(0.1), \
            "rebirth_1 magnitude = 1/10 = 0.1"
        assert out[V5_MAGNITUDE_BASE + 1] == pytest.approx(0.0)
        assert out[V5_MAGNITUDE_BASE + 2] == pytest.approx(0.0)

    def test_crime_and_punishment_magnitude(self):
        """crime_and_punishment_N -> N/10 at index 71. Grounded: effects.py:1178
        parse_crime_and_punishment uses re.match(r'crime_and_punishment_(\\d+)$', m)."""
        card = CardInstance(
            card_type=CardType.HERO,
            mana_cost=0,
            attack=0,
            hp=30,
            max_hp=30,
            mechanics=["crime_and_punishment_2"],
        )
        out = encode_card_shape_v5(card)
        assert out[V5_MAGNITUDE_BASE] == pytest.approx(0.0)
        assert out[V5_MAGNITUDE_BASE + 1] == pytest.approx(0.2), \
            "crime_and_punishment_2 magnitude = 2/10 = 0.2"
        assert out[V5_MAGNITUDE_BASE + 2] == pytest.approx(0.0)

    def test_target_ally_max_hp_plus_magnitude(self):
        """target_ally_max_hp_plus[_universal]_N -> N/10 at index 72.
        Grounded: effects.py:1143-1144 registers
        target_ally_max_hp_plus_{amount} / ..._universal_{amount}."""
        card = CardInstance(
            card_type=CardType.WARRIOR,
            mana_cost=2,
            attack=1,
            hp=2,
            max_hp=2,
            mechanics=["target_ally_max_hp_plus_universal_1"],
        )
        out = encode_card_shape_v5(card)
        assert out[V5_MAGNITUDE_BASE] == pytest.approx(0.0)
        assert out[V5_MAGNITUDE_BASE + 1] == pytest.approx(0.0)
        assert out[V5_MAGNITUDE_BASE + 2] == pytest.approx(0.1), \
            "target_ally_max_hp_plus magnitude = 1/10 = 0.1"

    def test_target_ally_max_hp_plus_non_universal_magnitude(self):
        """The non-universal variant also parses N."""
        card = CardInstance(
            card_type=CardType.WARRIOR,
            mechanics=["target_ally_max_hp_plus_3"],
        )
        out = encode_card_shape_v5(card)
        assert out[V5_MAGNITUDE_BASE + 2] == pytest.approx(0.3)

    def test_aoe_silence_has_no_magnitude(self):
        """aoe_silence carries NO magnitude (hardcoded limit=3 — effects.py:1020).
        Only the one-hot at 64 is set; magnitude scalars stay 0.0."""
        card = CardInstance(
            card_type=CardType.WARRIOR,
            mana_cost=7,
            attack=4,
            hp=5,
            max_hp=5,
            mechanics=["aoe_silence"],
        )
        out = encode_card_shape_v5(card)
        assert out[V5_NEW_ONEHOT_BASE] == pytest.approx(1.0), "aoe_silence one-hot at 64"
        assert out[V5_MAGNITUDE_BASE] == pytest.approx(0.0)
        assert out[V5_MAGNITUDE_BASE + 1] == pytest.approx(0.0)
        assert out[V5_MAGNITUDE_BASE + 2] == pytest.approx(0.0)

    def test_team_wide_shield_has_no_magnitude(self):
        """team_wide_shield carries NO magnitude (binary shield — effects.py:1071).
        Only the one-hot at 65 is set; magnitude scalars stay 0.0."""
        card = CardInstance(
            card_type=CardType.WARRIOR,
            mechanics=["team_wide_shield"],
        )
        out = encode_card_shape_v5(card)
        assert out[V5_NEW_ONEHOT_BASE + 1] == pytest.approx(1.0), \
            "team_wide_shield one-hot at 65"
        assert out[V5_MAGNITUDE_BASE] == pytest.approx(0.0)
        assert out[V5_MAGNITUDE_BASE + 1] == pytest.approx(0.0)
        assert out[V5_MAGNITUDE_BASE + 2] == pytest.approx(0.0)

    def test_magnitude_takes_max_across_multiple(self):
        """When multiple rebirth_N present, the max N wins (mirrors classic scalar
        max behavior and Rust card_shape_v5.rs max)."""
        card = CardInstance(
            card_type=CardType.WARRIOR,
            mechanics=["rebirth_1", "rebirth_8"],
        )
        out = encode_card_shape_v5(card)
        assert out[V5_MAGNITUDE_BASE] == pytest.approx(0.8), \
            "rebirth max(1,8)/10 = 0.8"

    def test_magnitude_is_clipped_to_1(self):
        """Magnitude > 10 clips to 1.0 (min(N/10, 1.0))."""
        card = CardInstance(
            card_type=CardType.WARRIOR,
            mechanics=["rebirth_15"],
        )
        out = encode_card_shape_v5(card)
        assert out[V5_MAGNITUDE_BASE] == pytest.approx(1.0), \
            "rebirth_15 magnitude clipped to 1.0"


# ============================================================================
# RUST <-> PYTHON 73-DIM PARITY (source-vs-source: mirror Rust unit assertions)
# ============================================================================

class TestRustPythonParity:
    """Mirror the exact assertions from ``card_shape_v5.rs`` unit tests in Python.
    Both encoders implement the same 73-dim layout; the Rust tests (which pass)
    assert these same float values.  Python producing the same values at the same
    indices establishes Rust<->Python parity at the card-shape level.  The full
    observation parity is validated transitively: Rust matches the golden
    fixtures (cargo golden_kernel tests), and the fixtures were regenerated from
    the Python encoder (source-vs-source oracle)."""

    def _warrior(self, mechanics: list[str], **kw) -> CardInstance:
        defaults = dict(
            card_type=CardType.WARRIOR, mana_cost=1, attack=1, hp=1, max_hp=1,
            level=1,
        )
        defaults.update(kw)
        defaults["mechanics"] = mechanics
        return CardInstance(**defaults)

    def test_rust_python_byte_identical_on_0_64(self):
        """Mirrors Rust v5_encoder_is_byte_identical_to_classic_on_0_64."""
        samples = [
            self._warrior(["battlecry_damage_1_random"], mana_cost=2, attack=2, hp=1),
            self._warrior(["aoe_silence"], mana_cost=7, attack=4, hp=5, max_hp=5),
            self._warrior(["rebirth_1"], mana_cost=8, attack=3, hp=7, max_hp=7),
            CardInstance(
                card_type=CardType.HERO, mana_cost=0, attack=0, hp=30, max_hp=30,
                level=1, mechanics=["crime_and_punishment_2"],
            ),
            self._warrior(["target_ally_max_hp_plus_universal_1"],
                          mana_cost=2, attack=1, hp=2, max_hp=2),
            self._warrior(["team_wide_shield"], mana_cost=7, attack=2, hp=4, max_hp=4),
            CardInstance(
                card_type=CardType.POTION, mana_cost=1, attack=0, hp=0, max_hp=0,
                level=1, mechanics=["desk_freeze", "damage_5"],
            ),
        ]
        for card in samples:
            v5 = encode_card_shape_v5(card, board_pos=0)
            classic = encode_card_shape(card, board_pos=0)
            np.testing.assert_array_equal(
                v5[:CARD_SHAPE_DIM], classic,
                err_msg=f"v5/classic [0:64) divergence for {card.mechanics}",
            )

    def test_rust_python_overlap_fix(self):
        """Mirrors Rust v5_overlap_fixed_no_index_collision."""
        card = CardInstance(
            card_type=CardType.POTION, mana_cost=1, attack=0, hp=0, max_hp=0,
            level=1, mechanics=["desk_freeze", "damage_5"],
        )
        out = encode_card_shape_v5(card)
        assert out[47] == pytest.approx(0.5)
        assert out[69] == pytest.approx(1.0)

    def test_rust_python_five_new_one_hots(self):
        """Mirrors Rust v5_five_new_mechanic_one_hots."""
        cases = [
            ("aoe_silence", 64),
            ("team_wide_shield", 65),
            ("rebirth_1", 66),
            ("crime_and_punishment_2", 67),
            ("target_ally_max_hp_plus_universal_1", 68),
        ]
        for mech, idx in cases:
            card = self._warrior([mech])
            out = encode_card_shape_v5(card)
            assert out[idx] == pytest.approx(1.0), f"flag at {idx} for {mech}"
            for i in range(64, 69):
                if i != idx:
                    assert out[i] == pytest.approx(0.0), \
                        f"other one-hot {i} must be 0 for {mech}"

    def test_rust_python_magnitude_scalars(self):
        """Mirrors Rust v5_magnitude_scalars_grounded."""
        # rebirth_1 -> 0.1 at 70
        out = encode_card_shape_v5(self._warrior(["rebirth_1"]))
        assert out[70] == pytest.approx(0.1)
        assert out[71] == pytest.approx(0.0)
        assert out[72] == pytest.approx(0.0)
        # crime_and_punishment_2 -> 0.2 at 71
        out = encode_card_shape_v5(CardInstance(
            card_type=CardType.HERO, mana_cost=0, attack=0, hp=30, max_hp=30,
            level=1, mechanics=["crime_and_punishment_2"],
        ))
        assert out[70] == pytest.approx(0.0)
        assert out[71] == pytest.approx(0.2)
        assert out[72] == pytest.approx(0.0)
        # target_ally_max_hp_plus_universal_1 -> 0.1 at 72
        out = encode_card_shape_v5(self._warrior(["target_ally_max_hp_plus_universal_1"]))
        assert out[70] == pytest.approx(0.0)
        assert out[71] == pytest.approx(0.0)
        assert out[72] == pytest.approx(0.1)
        # aoe_silence: one-hot at 64, no magnitude
        out = encode_card_shape_v5(self._warrior(["aoe_silence"]))
        assert out[64] == pytest.approx(1.0)
        assert out[70] == pytest.approx(0.0)
        assert out[71] == pytest.approx(0.0)
        assert out[72] == pytest.approx(0.0)


# ============================================================================
# DERIVED DIM CASCADE (contracts.py)
# ============================================================================

def test_persisted_history_card_dict_matches_live_card_instance():
    live = CardInstance(
        card_type=CardType.WARRIOR,
        mana_cost=3,
        attack=5,
        hp=4,
        max_hp=7,
        level=6,
        mechanics=["rebirth_1", "taunt"],
        is_ready=True,
        is_frozen=False,
    )
    persisted = {
        "card_type": "warrior",
        "mana_cost": 3,
        "attack": 5,
        "hp": 4,
        "max_hp": 7,
        "level": 6,
        "mechanics": ["rebirth_1", "taunt"],
        "is_ready": True,
        "is_frozen": False,
    }
    assert np.array_equal(
        encode_card_shape_v5(live),
        encode_card_shape_v5(persisted),
    )

class TestV5DimCascade:
    def test_dim_cascade_from_card_shape_v5(self):
        """All V5 dims derive from CARD_SHAPE_DIM_V5=73 (named offsets, not literals)."""
        from train_v3.contracts import (
            HISTORY_DIM,
            HISTORY_EVENT_DIM,
            HISTORY_EVENT_SOURCE_OFFSET,
            OBS_V1_DIM,
            OBS_V5_DIM,
            PRIVATE_CARD_SLOT_DIM,
            PRIVATE_INFO_DIM,
        )
        assert PRIVATE_CARD_SLOT_DIM == 1 + 1 + CARD_SHAPE_DIM_V5  # 75
        assert PRIVATE_INFO_DIM == 32 * PRIVATE_CARD_SLOT_DIM  # 2400
        assert HISTORY_EVENT_SOURCE_OFFSET == 16
        assert HISTORY_EVENT_DIM == HISTORY_EVENT_SOURCE_OFFSET + CARD_SHAPE_DIM_V5 * 2  # 162
        assert HISTORY_DIM == 20 * HISTORY_EVENT_DIM  # 3240
        assert OBS_V1_DIM == 1456  # frozen
        assert OBS_V5_DIM == OBS_V1_DIM + 32 + PRIVATE_INFO_DIM + HISTORY_DIM  # 7128
