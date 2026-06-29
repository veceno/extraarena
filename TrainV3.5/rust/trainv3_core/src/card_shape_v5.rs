//! V5 card-shape encoder — Block 0 bridge.
//!
//! ## Step-0 finding (Phase 8 investigation)
//!
//! Python `TrainV3.5/python/train_v3/obs_v5.py::encode_observation_v5` does NOT
//! define a separate V5 mechanics list. It imports and reuses the FROZEN
//! classic `ai/train_v2/classic_card_shape_v1.py::encode_card_shape` (64-dim,
//! 34 mechanic families from `core/state.py::MECHANICS_LIST`) for every card
//! slot — the V5 obs tensor, the private-info zone, and the history-event
//! source/target cards all use the classic 64-dim card shape. There is no
//! `MECHANICS_LIST_V5` / `card_shape_v5` symbol anywhere in the Python codebase
//! (grep confirms zero hits). The audit premise of "34 classic + 5 new families
//! = 39" does not match the current Python V5 encoder: V5 effectively uses the
//! 34-entry classic list. Count = 34.
//!
//! This module therefore exposes `MECHANICS_LIST_V5` mirroring Python's
//! `core.state.MECHANICS_LIST` (the 34 entries V5 actually consumes) and a V5
//! card-shape encoder that is byte-identical to the frozen classic encoder.
//! It is a dedicated Block 0 surface: future V5-only mechanic families (the
//! "5 new families" the audit anticipated) can be appended here WITHOUT
//! touching the frozen `card_shape.rs` (34) that backs V4-orig ONNX parity.
//!
//! ## index-47 overlap (frozen-classic, SHARED with Python — NOT fixed here)
//!
//! The classic 64-dim layout places 34 mechanic one-hot flags at indices
//! [14, 48) and 17 mechanic scalar channels at indices [47, 64). Index 47 is
//! BOTH the last mechanic one-hot (`desk_freeze`, `MECHANICS_LIST[33]` at
//! 14+33=47) AND the first scalar (`damage`, `scalars[0]`). This overlap is
//! present IDENTICALLY in Python `classic_card_shape_v1.py` (lines 142-143:
//! `out[14:14+len(flags)] = flags` → 14..48, `out[47:47+len(scalars)] =
//! scalars` → 47..64) and in Rust `card_shape.rs` (lines 75-86). Rust MATCHES
//! Python — there is no parity gap. Fixing it would break byte-parity with
//! Python AND the V4-orig ONNX model trained on the overlapping codec. It is a
//! frozen-classic bug requiring explicit user sign-off to change. This V5
//! module inherits the same layout to stay byte-identical to Python V5; the
//! overlap is documented, not fixed. See the Phase 8 report for escalation.
//!
//! ## 5-slot vs 7-slot obs divergence (frozen-classic violation — NOT fixed here)
//!
//! Separately, the classic OBSERVATION encoder (`classic_obs_v1`) diverges:
//! Rust `kernel.rs::encode_card_slots_v1` uses `0..7` board slots/side (hand at
//! slots 16-19), while current Python `classic_obs_v1._encode_card_slots` uses
//! `range(5)` board slots/side (hand at slots 12-15, 4 trailing zero slots).
//! Evidence: regenerating the `golden_trace_seed123` initial obs from current
//! Python yields sha256 `930174338105167b...`, but the committed fixture
//! records `7db0996fa0d7df6d...` (which Rust's 7-slot encoder reproduces). So
//! the 5 old frozen fixtures are STALE relative to current Python — current
//! Python `classic_obs_v1` was changed from 7-slot to 5-slot after the fixtures
//! were generated. This is a frozen-classic layout violation that affects V4-
//! orig ONNX semantic parity (hand cards land at different slot indices). It is
//! NOT in scope to fix without user sign-off. This module is card-shape only
//! (per-card 64-dim), which is byte-frozen and shared; the slot-layout
//! divergence lives in the observation encoder, not here.

use crate::card_shape::encode_card_shape as encode_card_shape_classic;
use crate::state::CardShapeInput;
use crate::CARD_SHAPE_DIM;

/// V5-effective mechanic family list. Mirrors Python `core.state.MECHANICS_LIST`
/// (34 entries) — the list Python `encode_observation_v5` actually consumes via
/// `classic_card_shape_v1.encode_card_shape`. Block 0 may append V5-only
/// families here without touching frozen `card_shape::MECHANICS_LIST`.
pub const MECHANICS_LIST_V5: [&str; 34] = [
    "taunt",
    "shield",
    "permanent_shield",
    "freeze",
    "aoe_freeze",
    "battlecry_freeze",
    "instant_kill",
    "bypass_taunt",
    "consume_ally",
    "reflect",
    "armor",
    "regen",
    "aura_atk",
    "cleave",
    "delete_target",
    "cast_random_spell",
    "choose_shield_damage",
    "start_mana",
    "mana_drain",
    "mana_gain",
    "battlecry_damage",
    "battlecry_heal",
    "battlecry_draw",
    "battlecry_buff",
    "damage",
    "heal",
    "damage_all",
    "heal_all",
    "buff_all",
    "summon",
    "deathrattle",
    "charge",
    "lifesteal",
    "desk_freeze",
];

/// V5 card-shape dimension. Same as the classic 64-dim shape (Python V5 reuses
/// `classic_card_shape_v1.CARD_SHAPE_DIM`). Block 0 may grow this when V5-only
/// families are appended.
pub const CARD_SHAPE_DIM_V5: usize = CARD_SHAPE_DIM;

/// Encode a card into the V5 card-feature vector.
///
/// Currently byte-identical to the frozen classic `encode_card_shape` (Python
/// V5 reuses the classic encoder). Kept as a distinct entry point so Block 0
/// can diverge the V5 shape without touching frozen classic.
pub fn encode_card_shape_v5(
    card: Option<&CardShapeInput<'_>>,
    board_pos: Option<usize>,
    hand_pos: Option<usize>,
    effective_attack: Option<i32>,
) -> [f32; CARD_SHAPE_DIM_V5] {
    encode_card_shape_classic(card, board_pos, hand_pos, effective_attack)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_shape::MECHANICS_LIST;
    use crate::state::CardType;

    #[test]
    fn v5_mechanics_list_length_matches_python_core_state() {
        // Python core.state.MECHANICS_LIST has 34 entries (counted from the
        // source). V5 reuses the classic 34-entry list — NOT 39.
        assert_eq!(MECHANICS_LIST_V5.len(), 34);
    }

    #[test]
    fn v5_mechanics_list_entries_are_unique() {
        let mut sorted = MECHANICS_LIST_V5;
        sorted.sort();
        for window in sorted.windows(2) {
            assert_ne!(window[0], window[1], "duplicate mechanic family");
        }
    }

    #[test]
    fn v5_mechanics_list_matches_frozen_classic_exactly() {
        // V5 currently reuses the frozen classic list verbatim (Python V5
        // imports classic_card_shape_v1.encode_card_shape which reads
        // core.state.MECHANICS_LIST).
        for (i, family) in MECHANICS_LIST.iter().enumerate() {
            assert_eq!(*family, MECHANICS_LIST_V5[i], "v5/classic mismatch at {i}");
        }
    }

    #[test]
    fn v5_encoder_is_byte_identical_to_classic_for_new_family_cards() {
        // Exercise the Phase 3-6 mechanic families (cards 15, 20, 23, 24, 25,
        // 47, 48, 50, 52) plus a vanilla warrior. The V5 encoder must produce
        // the exact same 64-float vector as the frozen classic encoder.
        let samples: Vec<CardShapeInput<'_>> = vec![
            CardShapeInput {
                card_type: CardType::Warrior,
                mana_cost: 2,
                attack: 2,
                hp: 1,
                max_hp: 1,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["battlecry_damage_1_random"],
            },
            CardShapeInput {
                card_type: CardType::Warrior,
                mana_cost: 3,
                attack: 2,
                hp: 2,
                max_hp: 2,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["consume_ally"],
            },
            CardShapeInput {
                card_type: CardType::Warrior,
                mana_cost: 7,
                attack: 7,
                hp: 7,
                max_hp: 7,
                is_ready: false,
                is_frozen: false,
                level: 5,
                mechanics: &["cleave_1_2"],
            },
            CardShapeInput {
                card_type: CardType::Warrior,
                mana_cost: 9,
                attack: 5,
                hp: 6,
                max_hp: 6,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["shield", "shield_refresh"],
            },
            CardShapeInput {
                card_type: CardType::Warrior,
                mana_cost: 4,
                attack: 10,
                hp: 10,
                max_hp: 10,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["instant_kill"],
            },
            CardShapeInput {
                card_type: CardType::Warrior,
                mana_cost: 7,
                attack: 4,
                hp: 5,
                max_hp: 5,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["aoe_silence"],
            },
            CardShapeInput {
                card_type: CardType::Warrior,
                mana_cost: 7,
                attack: 2,
                hp: 4,
                max_hp: 4,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["team_wide_shield"],
            },
            CardShapeInput {
                card_type: CardType::Warrior,
                mana_cost: 2,
                attack: 1,
                hp: 2,
                max_hp: 2,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["target_ally_max_hp_plus_universal_1"],
            },
            CardShapeInput {
                card_type: CardType::Warrior,
                mana_cost: 8,
                attack: 3,
                hp: 7,
                max_hp: 7,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["rebirth_1"],
            },
        ];
        for card in &samples {
            let v5 = encode_card_shape_v5(Some(card), Some(0), None, None);
            let classic = encode_card_shape_classic(Some(card), Some(0), None, None);
            assert_eq!(v5, classic, "v5/classic divergence for {:?}", card.mechanics);
        }
    }

    #[test]
    fn v5_encoder_handles_none_card() {
        let v5 = encode_card_shape_v5(None, None, None, None);
        assert_eq!(v5, [0.0_f32; CARD_SHAPE_DIM_V5]);
    }

    #[test]
    fn v5_mechanic_one_hot_region_layout_documented() {
        // The 34 one-hot flags occupy indices [14, 48); the 17 scalar channels
        // occupy [47, 64). Index 47 is shared (desk_freeze one-hot AND damage
        // scalar) — a frozen-classic overlap inherited from Python. This test
        // documents the layout boundary without asserting (broken) uniqueness
        // at index 47.
        // desk_freeze is MECHANICS_LIST_V5[33] → one-hot at 14 + 33 = 47.
        // damage_5 scalar (scalars[0]) ALSO writes index 47 — the shared
        // overlap. In the classic encoder the scalar block is written AFTER
        // the one-hot block (`out[47..64].copy_from_slice(&scalars)`), so for
        // a card carrying BOTH desk_freeze and damage_5 the scalar wins and
        // index 47 reads 0.5 (damage_5/10), NOT 1.0. This clobber is the
        // frozen-classic overlap bug shared identically with Python.
        let both = CardShapeInput {
            card_type: CardType::Potion,
            mana_cost: 1,
            attack: 0,
            hp: 0,
            max_hp: 0,
            is_ready: false,
            is_frozen: false,
            level: 1,
            mechanics: &["desk_freeze", "damage_5"],
        };
        let out_both = encode_card_shape_v5(Some(&both), None, None, None);
        assert_eq!(out_both[47], 0.5, "scalar clobbers one-hot at shared index 47");

        // desk_freeze ALONE: one-hot writes 1.0, scalar block writes 0.0 → 0.0 wins.
        let desk_only = CardShapeInput {
            card_type: CardType::Potion,
            mana_cost: 1,
            attack: 0,
            hp: 0,
            max_hp: 0,
            is_ready: false,
            is_frozen: false,
            level: 1,
            mechanics: &["desk_freeze"],
        };
        let out_desk = encode_card_shape_v5(Some(&desk_only), None, None, None);
        assert_eq!(out_desk[47], 0.0, "desk_freeze one-hot at 47 is clobbered to 0.0");

        // damage_5 ALONE: scalar writes 0.5 at index 47.
        let dmg_only = CardShapeInput {
            card_type: CardType::Potion,
            mana_cost: 1,
            attack: 0,
            hp: 0,
            max_hp: 0,
            is_ready: false,
            is_frozen: false,
            level: 1,
            mechanics: &["damage_5"],
        };
        let out_dmg = encode_card_shape_v5(Some(&dmg_only), None, None, None);
        assert_eq!(out_dmg[47], 0.5, "damage_5 scalar at index 47 (overlap target)");
        // The shared-write boundary is exactly index 47 = 14 + 33.
        assert_eq!(14 + 33, 47);
        assert_eq!(14 + MECHANICS_LIST_V5.len(), 48);
    }
}