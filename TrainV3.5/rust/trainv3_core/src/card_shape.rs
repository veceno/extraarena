use crate::state::{CardShapeInput, CardType};
use crate::CARD_SHAPE_DIM;

pub const MECHANICS_LIST: [&str; 34] = [
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

pub fn encode_card_shape(
    card: Option<&CardShapeInput<'_>>,
    board_pos: Option<usize>,
    hand_pos: Option<usize>,
    effective_attack: Option<i32>,
) -> [f32; CARD_SHAPE_DIM] {
    let mut out = [0.0_f32; CARD_SHAPE_DIM];
    let Some(card) = card else {
        return out;
    };

    match card.card_type {
        CardType::Hero => out[0] = 1.0,
        CardType::Warrior => out[1] = 1.0,
        CardType::Potion => out[2] = 1.0,
    }

    let eff_atk = effective_attack.unwrap_or(card.attack);
    out[3] = norm(card.mana_cost, 10.0);
    out[4] = norm(card.attack, 20.0);
    out[5] = norm(card.hp, 20.0);
    out[6] = norm(card.max_hp, 20.0);
    out[7] = card.hp_fraction();
    out[8] = if card.is_ready { 1.0 } else { 0.0 };
    out[9] = if card.is_frozen { 1.0 } else { 0.0 };
    out[10] = norm(card.level, 10.0);
    out[11] = norm(eff_atk, 20.0);
    out[12] = board_pos
        .map(|p| ((p as f64 + 1.0) / 8.0) as f32)
        .unwrap_or(0.0);
    out[13] = hand_pos
        .map(|p| ((p as f64 + 1.0) / 5.0) as f32)
        .unwrap_or(0.0);

    for (i, family) in MECHANICS_LIST.iter().enumerate() {
        if card
            .mechanics
            .iter()
            .any(|m| *m == *family || m.starts_with(&format!("{family}_")))
        {
            out[14 + i] = 1.0;
        }
    }

    let scalars = mechanic_scalars(card.mechanics);
    out[47..64].copy_from_slice(&scalars);
    out
}

fn norm(value: i32, divisor: f32) -> f32 {
    ((value as f64 / divisor as f64).min(1.0)) as f32
}

fn mechanic_scalars(mechanics: &[&str]) -> [f32; 17] {
    let mut out = [0.0_f32; 17];
    out[0] = scalar_damage_family(mechanics, 10.0);
    out[1] = scalar_max(mechanics, &["heal_", "heal_target_"], 10.0);
    out[2] = scalar_max(mechanics, &["aoe_damage_"], 10.0);
    out[3] = scalar_max(mechanics, &["battlecry_damage_"], 10.0);
    out[4] = scalar_max(
        mechanics,
        &[
            "battlecry_heal_",
            "battlecry_heal_hero_",
            "battlecry_heal_target_",
        ],
        10.0,
    );
    out[5] = scalar_max(
        mechanics,
        &["battlecry_buff_", "buff_all_", "buff_atk_"],
        10.0,
    );
    out[6] = scalar_second_after_prefix(mechanics, &["battlecry_buff_", "buff_all_"], 10.0);
    out[7] = scalar_max(mechanics, &["armor_"], 10.0);
    out[8] = scalar_max(mechanics, &["reflect_"], 10.0);
    out[9] = scalar_max(mechanics, &["regen_"], 10.0);
    out[10] = scalar_max(mechanics, &["aura_atk_"], 10.0);
    out[11] = scalar_max(mechanics, &["cleave_"], 10.0);
    out[12] = scalar_second_after_prefix(mechanics, &["cleave_"], 5.0);
    out[13] = scalar_max(mechanics, &["mana_gain_"], 10.0);
    out[14] = scalar_max(mechanics, &["mana_drain_"], 10.0);
    out[15] = scalar_max(mechanics, &["start_mana_"], 10.0);
    out[16] = if mechanics.iter().any(|m| *m == "summon") {
        1.0
    } else {
        scalar_max(mechanics, &["summon_"], 20.0)
    };
    out
}

fn scalar_damage_family(mechanics: &[&str], divisor: f32) -> f32 {
    let best = mechanics
        .iter()
        .filter_map(|m| number_after_damage_marker(m))
        .max()
        .unwrap_or(0);
    ((best as f64 / divisor as f64).min(1.0)) as f32
}

fn scalar_max(mechanics: &[&str], prefixes: &[&str], divisor: f32) -> f32 {
    let best = mechanics
        .iter()
        .filter_map(|m| first_number_after_prefix(m, prefixes))
        .max()
        .unwrap_or(0);
    ((best as f64 / divisor as f64).min(1.0)) as f32
}

fn scalar_second_after_prefix(mechanics: &[&str], prefixes: &[&str], divisor: f32) -> f32 {
    let best = mechanics
        .iter()
        .filter_map(|m| number_after_prefix(m, prefixes, 1))
        .max()
        .unwrap_or(0);
    (best as f32 / divisor).min(1.0)
}

fn first_number_after_prefix(value: &str, prefixes: &[&str]) -> Option<i32> {
    number_after_prefix(value, prefixes, 0)
}

fn number_after_damage_marker(value: &str) -> Option<i32> {
    let marker = "damage_";
    let start = if let Some(rest) = value.strip_prefix(marker) {
        rest
    } else {
        let idx = value.find("_damage_")?;
        &value[idx + "_damage_".len()..]
    };
    start.split('_').next()?.parse::<i32>().ok()
}

fn number_after_prefix(value: &str, prefixes: &[&str], index: usize) -> Option<i32> {
    for prefix in prefixes {
        if let Some(rest) = value.strip_prefix(prefix) {
            return rest.split('_').nth(index)?.parse::<i32>().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_basic_shape_and_mechanics() {
        let card = CardShapeInput {
            card_type: CardType::Warrior,
            mana_cost: 3,
            attack: 5,
            hp: 7,
            max_hp: 10,
            is_ready: true,
            is_frozen: false,
            level: 2,
            mechanics: &["taunt", "damage_5", "aura_atk_2"],
        };
        let out = encode_card_shape(Some(&card), Some(2), None, Some(7));
        assert_eq!(out[1], 1.0);
        assert_eq!(out[3], 0.3);
        assert_eq!(out[8], 1.0);
        assert_eq!(out[11], 0.35);
        assert_eq!(out[12], 0.375);
        assert_eq!(out[14], 1.0);
        assert_eq!(out[47], 0.5);
        assert_eq!(out[57], 0.2);
    }
}
