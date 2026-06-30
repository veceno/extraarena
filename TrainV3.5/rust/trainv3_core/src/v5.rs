use crate::{ACTION_FEATURE_DIM_V1, MAX_CANDIDATE_ACTIONS, OBS_DIM_V1};
use crate::card_shape_v5::CARD_SHAPE_DIM_V5;

pub const OWN_HAND_SLOTS: usize = 4;
pub const OWN_DECK_SLOTS: usize = 12;
pub const ENEMY_HAND_SLOTS: usize = 4;
pub const ENEMY_DECK_SLOTS: usize = 12;

// Per-card private-info slot = [occupied_flag, card_id_norm, card_shape_v5(73)].
pub const PRIVATE_CARD_SLOT_DIM: usize = 1 + 1 + CARD_SHAPE_DIM_V5; // 75
pub const PRIVATE_CARD_SLOTS: usize =
    OWN_HAND_SLOTS + OWN_DECK_SLOTS + ENEMY_HAND_SLOTS + ENEMY_DECK_SLOTS; // 32
pub const PRIVATE_INFO_DIM: usize = PRIVATE_CARD_SLOTS * PRIVATE_CARD_SLOT_DIM; // 2400

pub const V5_GLOBAL_DIM: usize = 32;
pub const HISTORY_EVENTS: usize = 20;
// History event = [13 metadata + 3 padding][source_card_v5][target_card_v5].
pub const HISTORY_EVENT_SOURCE_OFFSET: usize = 16;
pub const HISTORY_EVENT_DIM: usize = HISTORY_EVENT_SOURCE_OFFSET + CARD_SHAPE_DIM_V5 * 2; // 162
pub const HISTORY_DIM: usize = HISTORY_EVENTS * HISTORY_EVENT_DIM; // 3240
pub const OBS_DIM_V5: usize = OBS_DIM_V1 + V5_GLOBAL_DIM + PRIVATE_INFO_DIM + HISTORY_DIM; // 7128

#[derive(Debug, Clone, Copy)]
pub struct InfoModeV5 {
    pub adaptive_strength: f32,
    pub own_hand_identity_known: bool,
    pub own_deck_known: bool,
    pub enemy_hand_known: bool,
    pub enemy_deck_known: bool,
    pub enemy_deck_order_known: bool,
    pub draw_assist_enabled: bool,
    pub draw_assist_strength: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AssistModeV5 {
    pub assembler_enabled: bool,
    pub assembler_strength: f32,
    pub desirerer_enabled: bool,
    pub desirerer_strength: f32,
    pub teacher_hint_available: bool,
    pub assist_profile_id: i32,
}

impl Default for AssistModeV5 {
    fn default() -> Self {
        Self {
            assembler_enabled: false,
            assembler_strength: 0.0,
            desirerer_enabled: false,
            desirerer_strength: 0.0,
            teacher_hint_available: false,
            assist_profile_id: 0,
        }
    }
}

impl AssistModeV5 {
    pub fn clipped_assembler_strength(&self) -> f32 {
        if self.assembler_enabled {
            self.assembler_strength.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn clipped_desirerer_strength(&self) -> f32 {
        if self.desirerer_enabled {
            self.desirerer_strength.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn clipped_profile_id(&self) -> i32 {
        self.assist_profile_id.clamp(0, 15)
    }
}

impl Default for InfoModeV5 {
    fn default() -> Self {
        Self {
            adaptive_strength: 1.0,
            own_hand_identity_known: true,
            own_deck_known: true,
            enemy_hand_known: false,
            enemy_deck_known: false,
            enemy_deck_order_known: false,
            draw_assist_enabled: false,
            draw_assist_strength: 0.0,
        }
    }
}

impl InfoModeV5 {
    pub fn clipped_strength(&self) -> f32 {
        self.adaptive_strength.clamp(0.0, 1.0)
    }

    pub fn clipped_draw_assist_strength(&self) -> f32 {
        if self.draw_assist_enabled {
            self.draw_assist_strength.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveSearchCandidate {
    pub action_id: i32,
    pub policy_score: f32,
    pub rollout_score: f32,
}

pub fn adaptive_search_budget(
    search_candidates: usize,
    search_depth_plies: usize,
    adaptive_strength: f32,
    adaptive_runtime_enabled: bool,
) -> (usize, usize) {
    if !adaptive_runtime_enabled {
        return (search_candidates, search_depth_plies);
    }
    let strength = adaptive_strength.clamp(0.0, 1.0);
    let candidates = if search_candidates == 0 {
        0
    } else {
        ((search_candidates as f32) * strength).ceil().max(1.0) as usize
    };
    let depth = ((search_depth_plies as f32) * strength).ceil().max(0.0) as usize;
    (candidates, depth)
}

pub fn select_adaptive_search_action(
    candidates: &[AdaptiveSearchCandidate],
    adaptive_strength: f32,
) -> Option<i32> {
    if candidates.is_empty() {
        return None;
    }
    let strength = adaptive_strength.clamp(0.0, 1.0);
    let policy_values = normalize_scores(candidates.iter().map(|item| item.policy_score));
    let rollout_values = normalize_scores(candidates.iter().map(|item| item.rollout_score));
    let mut best_action = candidates[0].action_id;
    let mut best_score = f32::NEG_INFINITY;
    for (idx, item) in candidates.iter().enumerate() {
        let blended = (1.0 - strength) * policy_values[idx] + strength * rollout_values[idx];
        if blended > best_score
            || ((blended - best_score).abs() <= f32::EPSILON && item.action_id < best_action)
        {
            best_action = item.action_id;
            best_score = blended;
        }
    }
    Some(best_action)
}

fn normalize_scores(values: impl Iterator<Item = f32>) -> Vec<f32> {
    let raw: Vec<f32> = values.collect();
    let finite: Vec<f32> = raw.iter().copied().filter(|value| value.is_finite()).collect();
    if finite.is_empty() {
        return vec![0.0; raw.len()];
    }
    let lo = finite.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if hi <= lo {
        return raw
            .iter()
            .map(|value| if value.is_finite() { 0.5 } else { 0.0 })
            .collect();
    }
    raw.iter()
        .map(|value| {
            if value.is_finite() {
                (*value - lo) / (hi - lo)
            } else {
                0.0
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub struct V5TensorShapes {
    pub observation_dim: usize,
    pub action_count: usize,
    pub action_feature_dim: usize,
}

impl Default for V5TensorShapes {
    fn default() -> Self {
        Self {
            observation_dim: OBS_DIM_V5,
            action_count: MAX_CANDIDATE_ACTIONS,
            action_feature_dim: ACTION_FEATURE_DIM_V1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CARD_SHAPE_DIM;

    #[test]
    fn v5_dimension_contract_is_stable() {
        // Dim-73 cascade: CARD_SHAPE_DIM_V5=73 → PRIVATE_CARD_SLOT_DIM=75,
        // PRIVATE_INFO_DIM=2400, HISTORY_EVENT_DIM=162, HISTORY_DIM=3240,
        // OBS_DIM_V5=7128.  All derived from CARD_SHAPE_DIM_V5 (named offsets).
        assert_eq!(CARD_SHAPE_DIM, 64); // frozen classic stays 64
        assert_eq!(CARD_SHAPE_DIM_V5, 73);
        assert_eq!(PRIVATE_CARD_SLOT_DIM, 75);
        assert_eq!(PRIVATE_INFO_DIM, 2400);
        assert_eq!(HISTORY_EVENT_DIM, 162);
        assert_eq!(HISTORY_DIM, 3240);
        assert_eq!(OBS_DIM_V5, 7128);
        let shapes = V5TensorShapes::default();
        assert_eq!(shapes.observation_dim, 7128);
        assert_eq!(shapes.action_count, 601);
        assert_eq!(shapes.action_feature_dim, 171);
    }

    #[test]
    fn adaptive_strength_is_clipped() {
        assert_eq!(
            InfoModeV5 {
                adaptive_strength: -2.0,
                ..Default::default()
            }
            .clipped_strength(),
            0.0
        );
        assert_eq!(
            InfoModeV5 {
                adaptive_strength: 2.0,
                ..Default::default()
            }
            .clipped_strength(),
            1.0
        );
    }

    #[test]
    fn draw_assist_strength_is_zero_when_disabled_and_clipped_when_enabled() {
        assert_eq!(
            InfoModeV5 {
                draw_assist_enabled: false,
                draw_assist_strength: 0.8,
                ..Default::default()
            }
            .clipped_draw_assist_strength(),
            0.0
        );
        assert_eq!(
            InfoModeV5 {
                draw_assist_enabled: true,
                draw_assist_strength: 2.0,
                ..Default::default()
            }
            .clipped_draw_assist_strength(),
            1.0
        );
    }

    #[test]
    fn assist_mode_strengths_are_independent_and_clipped() {
        let assist = AssistModeV5 {
            assembler_enabled: true,
            assembler_strength: 2.0,
            desirerer_enabled: false,
            desirerer_strength: 0.8,
            teacher_hint_available: true,
            assist_profile_id: 99,
        };
        assert_eq!(assist.clipped_assembler_strength(), 1.0);
        assert_eq!(assist.clipped_desirerer_strength(), 0.0);
        assert_eq!(assist.clipped_profile_id(), 15);
    }

    #[test]
    fn adaptive_search_budget_scales_with_strength() {
        assert_eq!(adaptive_search_budget(8, 24, 0.25, true), (2, 6));
        assert_eq!(adaptive_search_budget(8, 24, 1.0, true), (8, 24));
        assert_eq!(adaptive_search_budget(8, 24, 0.25, false), (8, 24));
        assert_eq!(adaptive_search_budget(0, 24, 0.25, true), (0, 6));
    }

    #[test]
    fn adaptive_blended_search_selects_policy_low_strength_and_rollout_high_strength() {
        let candidates = [
            AdaptiveSearchCandidate {
                action_id: 10,
                policy_score: 1.0,
                rollout_score: 0.0,
            },
            AdaptiveSearchCandidate {
                action_id: 20,
                policy_score: 0.0,
                rollout_score: 1.0,
            },
        ];
        assert_eq!(select_adaptive_search_action(&candidates, 0.0), Some(10));
        assert_eq!(select_adaptive_search_action(&candidates, 0.25), Some(10));
        assert_eq!(select_adaptive_search_action(&candidates, 0.75), Some(20));
        assert_eq!(select_adaptive_search_action(&candidates, 1.0), Some(20));
    }
}
