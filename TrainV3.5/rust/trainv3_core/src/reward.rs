#[derive(Debug, Clone, Copy, Default)]
pub struct RewardSnapshot {
    pub my_hero_hp: i32,
    pub enemy_hero_hp: i32,
    pub my_board_count: i32,
    pub enemy_board_count: i32,
    pub my_mana: i32,
}

pub fn shaped_delta_reward(pre: RewardSnapshot, post: RewardSnapshot) -> f32 {
    let mut reward = 0.0_f32;
    let enemy_hp_delta = pre.enemy_hero_hp - post.enemy_hero_hp;
    if enemy_hp_delta > 0 {
        reward += 0.02 * enemy_hp_delta as f32;
    }
    let own_hp_delta = pre.my_hero_hp - post.my_hero_hp;
    if own_hp_delta > 0 {
        reward -= 0.01 * own_hp_delta as f32;
    }
    let enemy_killed = pre.enemy_board_count - post.enemy_board_count;
    if enemy_killed > 0 {
        reward += 0.03 * enemy_killed as f32;
    }
    let own_killed = pre.my_board_count - post.my_board_count;
    if own_killed > 0 {
        reward -= 0.02 * own_killed as f32;
    }
    let mana_spent = pre.my_mana - post.my_mana;
    if mana_spent > 0 {
        reward += (0.005 * mana_spent as f32).min(0.02);
    }
    reward
}

pub fn board_power(units: &[(i32, i32)]) -> i32 {
    units
        .iter()
        .map(|(attack, hp)| attack.max(&0) * hp.max(&0))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaped_reward_matches_trainv2_coefficients() {
        let pre = RewardSnapshot {
            my_hero_hp: 30,
            enemy_hero_hp: 30,
            my_board_count: 2,
            enemy_board_count: 2,
            my_mana: 5,
        };
        let post = RewardSnapshot {
            my_hero_hp: 28,
            enemy_hero_hp: 25,
            my_board_count: 1,
            enemy_board_count: 1,
            my_mana: 2,
        };
        let reward = shaped_delta_reward(pre, post);
        assert!((reward - 0.105).abs() < 1e-6);
    }
}
