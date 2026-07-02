use crate::{ACTION_FEATURE_DIM_V1, MAX_CANDIDATE_ACTIONS, OBS_DIM_V1};

#[derive(Debug, Clone, Copy)]
pub struct TrainEnvConfig {
    pub verify_mask: bool,
    pub include_preview_features: bool,
    pub placement_append_only: bool,
    pub mana_per_turn: i32,
}

impl Default for TrainEnvConfig {
    fn default() -> Self {
        Self {
            verify_mask: false,
            include_preview_features: false,
            placement_append_only: true,
            mana_per_turn: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub observation: Vec<f32>,
    pub action_mask: Vec<f32>,
    pub action_features: Vec<f32>,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

impl StepOutput {
    pub fn zeroed() -> Self {
        Self {
            observation: vec![0.0; OBS_DIM_V1],
            action_mask: vec![0.0; MAX_CANDIDATE_ACTIONS],
            action_features: vec![0.0; MAX_CANDIDATE_ACTIONS * ACTION_FEATURE_DIM_V1],
            reward: 0.0,
            terminated: false,
            truncated: false,
        }
    }
}
