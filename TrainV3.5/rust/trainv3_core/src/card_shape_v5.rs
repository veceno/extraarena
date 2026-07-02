//! V5 card-shape encoder — Block 0 fork of the frozen classic `card_shape.rs`.
//!
//! ## Layout (73 floats, grow + disjoint from classic 64)
//!
//! ```text
//! [0:3]    Type one-hot: hero, warrior, potion
//! [3:14]   Numeric/status (mana_cost, attack, hp, max_hp, hp_fraction,
//!          is_ready, is_frozen, level, effective_attack, board_pos, hand_pos)
//! [14:47]  33 classic mechanic one-hots (MECHANICS_LIST[0:33] — first 33;
//!          desk_freeze, MECHANICS_LIST[33], is NOT written here — see fix at 69)
//! [47:64]  17 classic mechanic scalar channels — byte-identical to frozen classic
//! [64:69]  5 NEW V5-only mechanic one-hots:
//!            [64] aoe_silence
//!            [65] team_wide_shield
//!            [66] rebirth               (rebirth_N)
//!            [67] crime_and_punishment   (crime_and_punishment_N)
//!            [68] target_ally_max_hp_plus (target_ally_max_hp_plus[_universal]_N)
//! [69]     desk_freeze one-hot — index-47-overlap FIX.  The frozen classic
//!          writes desk_freeze's flag at index 47 where it is ALWAYS clobbered
//!          by the damage scalar.  V5 re-encodes it here at a disjoint index.
//! [70:73]  3 NEW V5 magnitude scalars (families with a parsed N parameter):
//!            [70] rebirth_N / 10.0
//!            [71] crime_and_punishment_N / 10.0
//!            [72] target_ally_max_hp_plus_N / 10.0
//!          aoe_silence and team_wide_shield carry NO magnitude (hardcoded
//!          limit=3 / binary shield — core/effects.py:1020-1032, 1071-1086).
//! ```
//!
//! [0:64) is byte-identical to the frozen classic encoder (it calls
//! `encode_card_shape_classic` and copies the result).  The classic encoder
//! ALWAYS overwrites index 47 with the damage scalar, so the desk_freeze flag
//! written there is invisible in the classic output — V5 simply re-encodes
//! desk_freeze at index 69, which leaves [0:64) unchanged.
//!
//! ## Frozen guard
//!
//! `card_shape.rs` (the 34-entry MECHANICS_LIST + 64-dim encoder) backs V4-orig
//! ONNX and is NEVER modified.  V5-only families live here in this separate
//! module.  `core/state.py:MECHANICS_LIST` stays 34 — the 5 new families are
//! V5-fork one-hots, NOT added to the frozen classic list.

use crate::card_shape::encode_card_shape as encode_card_shape_classic;
use crate::state::CardShapeInput;
use crate::CARD_SHAPE_DIM;

/// V5-effective mechanic family list. Mirrors Python `core.state.MECHANICS_LIST`
/// (34 entries) — the list the frozen classic encoder consumes.  V5 reuses
/// these 34 for [0:64) and appends V5-only families below.
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

/// 5 NEW V5-only mechanic families (NOT in the frozen 34-entry list).
/// Grounded in core/effects.py.
pub const V5_NEW_MECHANIC_FAMILIES: [&str; 5] = [
    "aoe_silence",            // effects.py:1020 — no _N (hardcoded limit=3)
    "team_wide_shield",       // effects.py:1071 — no _N (binary shield, :1055)
    "rebirth",                // effects.py:1161 — rebirth_N (parse_rebirth)
    "crime_and_punishment",   // effects.py:1175 — crime_and_punishment_N
    "target_ally_max_hp_plus", // effects.py:1112 — target_ally_max_hp_plus[_universal]_N
];

/// Families that carry a parsed magnitude N.  Only these get a magnitude scalar;
/// aoe_silence and team_wide_shield do not.
pub const V5_MAGNITUDE_FAMILIES: [&str; 3] = [
    "rebirth",
    "crime_and_punishment",
    "target_ally_max_hp_plus",
];

pub const V5_NEW_ONEHOT_BASE: usize = 64;
pub const V5_DESK_FREEZE_INDEX: usize = 69;
pub const V5_MAGNITUDE_BASE: usize = 70;
pub const V5_MAGNITUDE_NORMALIZER: f32 = 10.0;

/// V5 card-shape dimension.  Grow + disjoint: 64 classic + 5 new one-hots +
/// 1 desk_freeze fix + 3 magnitude scalars = 73.
pub const CARD_SHAPE_DIM_V5: usize = 73;

/// Encode a card into the V5 73-float card-feature vector.
///
/// [0:64) is byte-identical to the frozen classic `encode_card_shape` (it calls
/// the classic encoder and copies).  [64:73) appends V5-only one-hots, the
/// desk_freeze overlap-fix flag, and grounded magnitude scalars.
pub fn encode_card_shape_v5(
    card: Option<&CardShapeInput<'_>>,
    board_pos: Option<usize>,
    hand_pos: Option<usize>,
    effective_attack: Option<i32>,
) -> [f32; CARD_SHAPE_DIM_V5] {
    // [0:64) — byte-identical to frozen classic.
    let classic = encode_card_shape_classic(card, board_pos, hand_pos, effective_attack);
    let mut out = [0.0_f32; CARD_SHAPE_DIM_V5];
    out[..CARD_SHAPE_DIM].copy_from_slice(&classic);

    let Some(card) = card else {
        return out;
    };

    // --- [64:69) 5 new mechanic one-hots ---
    for (i, family) in V5_NEW_MECHANIC_FAMILIES.iter().enumerate() {
        let prefix = format!("{family}_");
        if card
            .mechanics
            .iter()
            .any(|m| *m == *family || m.starts_with(&prefix))
        {
            out[V5_NEW_ONEHOT_BASE + i] = 1.0;
        }
    }

    // --- [69] desk_freeze overlap-fix one-hot ---
    // Classic writes desk_freeze's flag at index 47 (MECHANICS_LIST[33] → 14+33)
    // where it is ALWAYS clobbered by the damage scalar.  V5 re-encodes it here.
    if card
        .mechanics
        .iter()
        .any(|m| *m == "desk_freeze" || m.starts_with("desk_freeze_"))
    {
        out[V5_DESK_FREEZE_INDEX] = 1.0;
    }

    // --- [70:73) 3 magnitude scalars (grounded in core/effects.py) ---
    // rebirth_N             → effects.py:1164 parse_rebirth
    // crime_and_punishment_N → effects.py:1178 parse_crime_and_punishment
    // target_ally_max_hp_plus[_universal]_N → effects.py:1143-1144
    for (i, family) in V5_MAGNITUDE_FAMILIES.iter().enumerate() {
        let prefix = format!("{family}_");
        let best = card
            .mechanics
            .iter()
            .filter_map(|m| {
                m.strip_prefix(&prefix).and_then(|rest| {
                    // rest is "N" or "universal_N" (target_ally_max_hp_plus).
                    // Parse the last '_'-separated token as the magnitude.
                    rest.rsplit('_').next().and_then(|tok| tok.parse::<i32>().ok())
                })
            })
            .max()
            .unwrap_or(0);
        out[V5_MAGNITUDE_BASE + i] = ((best as f64 / V5_MAGNITUDE_NORMALIZER as f64).min(1.0)) as f32;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card_shape::MECHANICS_LIST;
    use crate::state::CardType;

    #[test]
    fn v5_mechanics_list_length_matches_python_core_state() {
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
        for (i, family) in MECHANICS_LIST.iter().enumerate() {
            assert_eq!(*family, MECHANICS_LIST_V5[i], "v5/classic mismatch at {i}");
        }
    }

    #[test]
    fn v5_card_shape_dim_is_73() {
        assert_eq!(CARD_SHAPE_DIM_V5, 73);
    }

    #[test]
    fn v5_encoder_handles_none_card() {
        let v5 = encode_card_shape_v5(None, None, None, None);
        assert_eq!(v5, [0.0_f32; CARD_SHAPE_DIM_V5]);
    }

    #[test]
    fn v5_encoder_is_byte_identical_to_classic_on_0_64() {
        // The V5 encoder [0:64) must be byte-identical to the frozen classic
        // encoder — the warm-start safety.  Verified across representative cards.
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
                mana_cost: 8,
                attack: 3,
                hp: 7,
                max_hp: 7,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["rebirth_1"],
            },
            CardShapeInput {
                card_type: CardType::Hero,
                mana_cost: 0,
                attack: 0,
                hp: 30,
                max_hp: 30,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["crime_and_punishment_2"],
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
                card_type: CardType::Potion,
                mana_cost: 1,
                attack: 0,
                hp: 0,
                max_hp: 0,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &["desk_freeze", "damage_5"],
            },
        ];
        for card in &samples {
            let v5 = encode_card_shape_v5(Some(card), Some(0), None, None);
            let classic = encode_card_shape_classic(Some(card), Some(0), None, None);
            assert_eq!(
                &v5[..CARD_SHAPE_DIM],
                &classic[..],
                "v5/classic [0:64) divergence for {:?}",
                card.mechanics
            );
        }
    }

    #[test]
    fn v5_overlap_fixed_no_index_collision() {
        // A card carrying desk_freeze + damage_5 must encode BOTH at disjoint
        // indices — the damage scalar at 47 (classic) and the desk_freeze
        // one-hot at 69 (V5 fix).  Neither clobbers the other.
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
        let out = encode_card_shape_v5(Some(&both), None, None, None);
        // damage_5 scalar at classic index 47 = 0.5 (NOT clobbered by desk_freeze).
        assert_eq!(out[47], 0.5, "damage scalar at index 47");
        // desk_freeze one-hot at V5 index 69 = 1.0 (NOT clobbered by damage).
        assert_eq!(out[69], 1.0, "desk_freeze one-hot at disjoint index 69");
        // Both features survive at disjoint indices.
        assert_ne!(out[47], 0.0);
        assert_ne!(out[69], 0.0);
    }

    #[test]
    fn v5_desk_freeze_alone_is_fixed_not_lost() {
        // Classic clobbers desk_freeze at 47 to 0.0; V5 re-encodes at 69.
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
        let out = encode_card_shape_v5(Some(&desk_only), None, None, None);
        assert_eq!(out[47], 0.0, "classic index 47 = 0.0 (clobbered, as in classic)");
        assert_eq!(out[69], 1.0, "V5 index 69 = 1.0 (desk_freeze fixed)");
    }

    #[test]
    fn v5_five_new_mechanic_one_hots() {
        // Each new family sets its flag at [64:69) and leaves the others 0.0.
        let cases: [(&str, usize); 5] = [
            ("aoe_silence", 64),
            ("team_wide_shield", 65),
            ("rebirth_1", 66),
            ("crime_and_punishment_2", 67),
            ("target_ally_max_hp_plus_universal_1", 68),
        ];
        for (mech, expected_idx) in cases {
            let card = CardShapeInput {
                card_type: CardType::Warrior,
                mana_cost: 1,
                attack: 1,
                hp: 1,
                max_hp: 1,
                is_ready: false,
                is_frozen: false,
                level: 1,
                mechanics: &[mech],
            };
            let out = encode_card_shape_v5(Some(&card), None, None, None);
            assert_eq!(
                out[expected_idx], 1.0,
                "family flag at {expected_idx} for {mech}"
            );
            // The other 4 new one-hots must be 0.0.
            for i in 64..69 {
                if i != expected_idx {
                    assert_eq!(out[i], 0.0, "other new one-hot at {i} must be 0.0 for {mech}");
                }
            }
        }
    }

    #[test]
    fn v5_magnitude_scalars_grounded() {
        // rebirth_1 → 1/10 = 0.1 at index 70
        let rebirth = CardShapeInput {
            card_type: CardType::Warrior,
            mana_cost: 8,
            attack: 3,
            hp: 7,
            max_hp: 7,
            is_ready: false,
            is_frozen: false,
            level: 1,
            mechanics: &["rebirth_1"],
        };
        let out = encode_card_shape_v5(Some(&rebirth), None, None, None);
        assert_eq!(out[70], 0.1, "rebirth_1 magnitude = 1/10 = 0.1");
        assert_eq!(out[71], 0.0, "crime_and_punishment magnitude absent");
        assert_eq!(out[72], 0.0, "target_ally_max_hp_plus magnitude absent");

        // crime_and_punishment_2 → 2/10 = 0.2 at index 71
        let crime = CardShapeInput {
            card_type: CardType::Hero,
            mana_cost: 0,
            attack: 0,
            hp: 30,
            max_hp: 30,
            is_ready: false,
            is_frozen: false,
            level: 1,
            mechanics: &["crime_and_punishment_2"],
        };
        let out = encode_card_shape_v5(Some(&crime), None, None, None);
        assert_eq!(out[70], 0.0, "rebirth magnitude absent");
        assert_eq!(out[71], 0.2, "crime_and_punishment_2 magnitude = 2/10 = 0.2");
        assert_eq!(out[72], 0.0, "target_ally_max_hp_plus magnitude absent");

        // target_ally_max_hp_plus_universal_1 → 1/10 = 0.1 at index 72
        let hpplus = CardShapeInput {
            card_type: CardType::Warrior,
            mana_cost: 2,
            attack: 1,
            hp: 2,
            max_hp: 2,
            is_ready: false,
            is_frozen: false,
            level: 1,
            mechanics: &["target_ally_max_hp_plus_universal_1"],
        };
        let out = encode_card_shape_v5(Some(&hpplus), None, None, None);
        assert_eq!(out[70], 0.0, "rebirth magnitude absent");
        assert_eq!(out[71], 0.0, "crime_and_punishment magnitude absent");
        assert_eq!(out[72], 0.1, "target_ally_max_hp_plus magnitude = 1/10 = 0.1");

        // aoe_silence and team_wide_shield have NO magnitude scalar.
        let aoe = CardShapeInput {
            card_type: CardType::Warrior,
            mana_cost: 7,
            attack: 4,
            hp: 5,
            max_hp: 5,
            is_ready: false,
            is_frozen: false,
            level: 1,
            mechanics: &["aoe_silence"],
        };
        let out = encode_card_shape_v5(Some(&aoe), None, None, None);
        assert_eq!(out[64], 1.0, "aoe_silence one-hot at 64");
        assert_eq!(out[70], 0.0, "aoe_silence has no magnitude (scalar 70 = 0.0)");
        assert_eq!(out[71], 0.0);
        assert_eq!(out[72], 0.0);
    }
}