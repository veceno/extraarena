use rand::seq::SliceRandom;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;

use crate::action_codec::{
    decode_action_id, CandidateAction, ATTACK_BASE, NUM_ATTACK_TARGETS, NUM_BOARD, NUM_HAND,
    NUM_PLAY_POS, NUM_PLAY_TARGETS, PLAY_BASE, PLAY_STRIDE,
};
use crate::card_shape::{encode_card_shape, MECHANICS_LIST};
use crate::state::{CardShapeInput, CardType};
use crate::v5::{
    AssistModeV5, InfoModeV5, ENEMY_DECK_SLOTS, ENEMY_HAND_SLOTS, HISTORY_DIM, HISTORY_EVENTS,
    HISTORY_EVENT_DIM, OBS_DIM_V5, OWN_DECK_SLOTS, OWN_HAND_SLOTS, PRIVATE_CARD_SLOT_DIM,
    PRIVATE_INFO_DIM, V5_GLOBAL_DIM,
};
use crate::{ACTION_FEATURE_DIM_V1, CARD_SHAPE_DIM, MAX_CANDIDATE_ACTIONS, OBS_DIM_V1};

const GLOBAL_DIM_V1: usize = 32;
const ZONE_SUMMARY_DIM_V1: usize = 48;
const CARD_ID_NORMALIZER: f32 = 1000.0;

// Weighted-draw constants — mirror core/engine.py (Phase 1 parity port).
const HAND_CAP: usize = 4;
const STUCK_BONUS: f32 = 0.5;
const COST_BIAS: f32 = 0.3;
const CHEAP_COST_MAX: i32 = 2;
const EXPENSIVE_COST_MIN: i32 = 4;

// Game-rule board cap — mirrors core/engine.py `len(player.board) >= 5`
// (the board-full guard for playing warriors, hardcoded 5 in Python). This
// is the GAME RULE, distinct from `NUM_BOARD=7` which is the frozen
// classic_actions_v1 codec capacity (obs/action slot LAYOUT stays 7).
// Phase 2: gap AC-FFI-1.
pub const GAME_BOARD_CAP: usize = 5;

// Mana-draw cost base — mirrors core/engine.py `MANA_DRAW_BASE = 2`.
// N-th mana draw in a turn costs `MANA_DRAW_BASE * (count + 1)` (2, 4, 6, ...).
const MANA_DRAW_BASE: i32 = 2;

/// Recorded-outcome RNG for draws (task #14, closes Phase-1 gap DW-7 for
/// multi-card decks).
///
/// **Design = outcome-based, NOT RNG-primitive-based.** Instead of replaying
/// Python's MT19937 `getrandbits`/`next_u32` sequence (rand's `gen_range`
/// internals don't align with Python's `_randbelow`), Rust records the
/// OUTCOMES of RNG decisions in Python (`golden_trace.py` monkeypatches
/// `_weighted_choice_idx` + the env RNG's `shuffle`) and applies them
/// directly here.
///
/// Two outcome streams are snapshotted per step (so Rust replays per-step in
/// call order):
///   * `picks`  — the chosen deck index (usize) for every
///     `_weighted_choice_idx` call (clean draw + overdraw-discard).
///   * `orders` — the post-shuffle deck `card_id` sequence for every
///     graveyard→deck reshuffle.
///
/// `Live` wraps any `RngCore` (training / FFI path: `WorkerRng` or `ChaCha`).
/// `Recorded` is built from a fixture step's `draw_picks` + `reshuffle_orders`
/// for golden-trace replay. The protocol is extensible: later phases add
/// armor/card15 RNG as new outcome streams on this enum.
pub enum DrawRng<'a> {
    Live(&'a mut dyn RngCore),
    Recorded {
        picks: VecDeque<usize>,
        orders: VecDeque<Vec<i32>>,
    },
}

impl<'a> DrawRng<'a> {
    /// Wrap a live `RngCore` (training / FFI). The frozen 601 codec and all
    /// non-draw RNG paths are unaffected. Accepts `&mut dyn RngCore` so callers
    /// pass `&mut WorkerRng` / `&mut ChaChaRng` (unsized coercion).
    pub fn live(rng: &'a mut dyn RngCore) -> Self {
        DrawRng::Live(rng)
    }

    /// Build a `Recorded` DrawRng from a fixture step's recorded outcomes.
    pub fn recorded(picks: Vec<usize>, orders: Vec<Vec<i32>>) -> Self {
        DrawRng::Recorded {
            picks: picks.into(),
            orders: orders.into(),
        }
    }

    /// Pop the next recorded draw-pick index. Falls back to `0` (with a
    /// debug warning) if the recorded stream is exhausted — this keeps
    /// replay robust to a draw-pick/order desync instead of panicking, and
    /// `0` matches the `Deterministic` zero-RNG behaviour for single-card
    /// decks.
    fn next_pick(&mut self) -> usize {
        match self {
            DrawRng::Recorded { picks, .. } => picks.pop_front().unwrap_or_else(|| {
                eprintln!("[DrawRng] recorded pick stream exhausted; falling back to 0");
                0
            }),
            DrawRng::Live(_) => unreachable!("next_pick called on Live DrawRng"),
        }
    }

    /// Pop the next recorded reshuffle order (post-shuffle deck card_ids).
    fn next_order(&mut self) -> Option<Vec<i32>> {
        match self {
            DrawRng::Recorded { orders, .. } => orders.pop_front(),
            DrawRng::Live(_) => unreachable!("next_order called on Live DrawRng"),
        }
    }

    fn is_recorded(&self) -> bool {
        matches!(self, DrawRng::Recorded { .. })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GoldenTrace {
    pub schema: String,
    pub env_config: GoldenTraceConfig,
    pub initial: GoldenSnapshot,
    pub steps: Vec<GoldenStep>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GoldenTraceConfig {
    #[serde(default)]
    pub seed: i32,
    #[serde(default)]
    pub verify_mask: bool,
    pub placement_mode: String,
    #[serde(default)]
    pub include_preview: bool,
    #[serde(default)]
    pub adaptive_strength: f32,
    #[serde(default = "default_true")]
    pub own_hand_identity_known: bool,
    #[serde(default = "default_true")]
    pub own_deck_known: bool,
    #[serde(default = "default_true")]
    pub enemy_hand_known: bool,
    #[serde(default = "default_true")]
    pub enemy_deck_known: bool,
    #[serde(default)]
    pub enemy_deck_order_known: bool,
    #[serde(default)]
    pub draw_assist_enabled: bool,
    #[serde(default)]
    pub draw_assist_strength: f32,
    #[serde(default)]
    pub assembler_enabled: bool,
    #[serde(default)]
    pub assembler_strength: f32,
    #[serde(default)]
    pub desirerer_enabled: bool,
    #[serde(default)]
    pub desirerer_strength: f32,
    #[serde(default)]
    pub teacher_hint_available: bool,
    #[serde(default)]
    pub assist_profile_id: i32,
    #[serde(default)]
    pub v5_weighted_reward: bool,
    #[serde(default = "default_mana_per_turn")]
    pub mana_per_turn: i32,
    #[serde(default)]
    pub overdraw_to_discard: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GoldenSnapshot {
    pub state: KernelState,
    #[serde(default)]
    pub history_events: Vec<KernelHistoryEvent>,
    pub state_sha256: String,
    pub obs_sha256_f32_le: String,
    pub mask_sha256_f32_le: String,
    pub action_features_sha256_f32_le: String,
    pub obs_v5_sha256_f32_le: Option<String>,
    #[serde(default)]
    pub legal_ids: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GoldenStep {
    pub t: usize,
    pub acting_player_id: i32,
    pub action_id: usize,
    #[serde(default)]
    pub reward: f32,
    #[serde(default)]
    pub terminated: bool,
    #[serde(default)]
    pub truncated: bool,
    pub pre: GoldenSnapshot,
    pub post: GoldenSnapshot,
    pub reward_components_v5: RewardComponentsV5,
    /// Parallel binary mana_draw head (Phase 2: MD-3). Whether mana_draw was
    /// a legal action for the acting player at this step (recorded from
    /// Python `legal_actions`). Default false for pre-Phase-2 fixtures.
    #[serde(default)]
    pub mana_draw_legal: bool,
    /// Whether this step's action was a mana_draw (the parallel head was
    /// taken instead of decoding `action_id`). Default false. Rust replay
    /// passes this as the `mana_draw_flag` argument to `apply_action`.
    #[serde(default)]
    pub mana_draw_taken: bool,
    /// Recorded-outcome RNG (task #14 / DW-7): the deck index chosen by every
    /// `_weighted_choice_idx` call during this step (clean draw +
    /// overdraw-discard). Rust replay pops these in call order. Default empty
    /// for pre-#14 fixtures (single-card decks where `Deterministic` idx 0
    /// suffices).
    #[serde(default)]
    pub draw_picks: Vec<usize>,
    /// Recorded-outcome RNG: the post-shuffle deck `card_id` sequence for every
    /// graveyard→deck reshuffle during this step. Rust replay reorders the
    /// graveyard to match. Default empty.
    #[serde(default)]
    pub reshuffle_orders: Vec<Vec<i32>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KernelState {
    pub current_turn_owner_id: i32,
    pub turn_number: i32,
    pub status: String,
    pub p1: KernelPlayer,
    pub p2: KernelPlayer,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KernelPlayer {
    pub user_id: i32,
    pub mana: i32,
    pub max_mana: i32,
    pub hero: KernelCard,
    #[serde(default)]
    pub hand: Vec<KernelCard>,
    #[serde(default)]
    pub board: Vec<KernelCard>,
    #[serde(default)]
    pub deck: Vec<KernelCard>,
    #[serde(default)]
    pub graveyard: Vec<KernelCard>,
    #[serde(default)]
    pub trophies: i32,
    #[serde(default)]
    pub mana_draw_count_this_turn: i32,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct KernelCard {
    pub card_id: i32,
    #[serde(rename = "type")]
    pub card_type: String,
    pub mana_cost: i32,
    pub attack: i32,
    pub hp: i32,
    pub max_hp: i32,
    #[serde(default)]
    pub mechanics: Vec<String>,
    #[serde(default)]
    pub is_ready: bool,
    #[serde(default)]
    pub is_frozen: bool,
    #[serde(default)]
    pub level: i32,
    #[serde(default)]
    pub skip_count: i32,
    #[serde(default)]
    pub base_attack: i32,
    #[serde(default)]
    pub base_hp: i32,
    #[serde(default)]
    pub base_max_hp: i32,
    #[serde(default)]
    pub base_mana_cost: i32,
    #[serde(default)]
    pub base_mechanics: Vec<String>,
    #[serde(default)]
    pub base_snapshot_set: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KernelHistoryEvent {
    #[serde(default)]
    pub actor_id: i32,
    #[serde(default)]
    pub action_id: usize,
    #[serde(default)]
    pub action_type: String,
    #[serde(default)]
    pub enemy_hero_hp_delta: i32,
    #[serde(default)]
    pub own_hero_hp_delta: i32,
    #[serde(default)]
    pub my_board_count_delta: i32,
    #[serde(default)]
    pub enemy_board_count_delta: i32,
    #[serde(default)]
    pub board_power_delta: f32,
    #[serde(default)]
    pub turn_number: i32,
    #[serde(default)]
    pub source_card: Option<KernelCard>,
    #[serde(default)]
    pub target_card: Option<KernelCard>,
}

#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    pub placement_mode: PlacementMode,
    pub include_preview: bool,
    pub info_mode: InfoModeV5,
    pub assist_mode: AssistModeV5,
    pub mana_per_turn: i32,
    pub v5_weighted_reward: bool,
    pub overdraw_to_discard: bool,
    pub seed: u64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            placement_mode: PlacementMode::AppendOnly,
            include_preview: false,
            info_mode: InfoModeV5::default(),
            assist_mode: AssistModeV5::default(),
            mana_per_turn: 1,
            v5_weighted_reward: false,
            overdraw_to_discard: false,
            seed: 0,
        }
    }
}

impl KernelConfig {
    pub fn from_trace_config(config: &GoldenTraceConfig) -> Self {
        Self {
            placement_mode: PlacementMode::from_str(&config.placement_mode),
            include_preview: config.include_preview,
            info_mode: InfoModeV5 {
                adaptive_strength: config.adaptive_strength,
                own_hand_identity_known: config.own_hand_identity_known,
                own_deck_known: config.own_deck_known,
                enemy_hand_known: config.enemy_hand_known,
                enemy_deck_known: config.enemy_deck_known,
                enemy_deck_order_known: config.enemy_deck_order_known,
                draw_assist_enabled: config.draw_assist_enabled,
                draw_assist_strength: config.draw_assist_strength,
                ..InfoModeV5::default()
            },
            assist_mode: AssistModeV5 {
                assembler_enabled: config.assembler_enabled,
                assembler_strength: config.assembler_strength,
                desirerer_enabled: config.desirerer_enabled,
                desirerer_strength: config.desirerer_strength,
                teacher_hint_available: config.teacher_hint_available,
                assist_profile_id: config.assist_profile_id,
            },
            mana_per_turn: config.mana_per_turn.max(1),
            v5_weighted_reward: config.v5_weighted_reward,
            overdraw_to_discard: config.overdraw_to_discard,
            seed: config.seed.max(0) as u64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementMode {
    Full,
    AppendOnly,
}

impl PlacementMode {
    fn from_str(value: &str) -> Self {
        match value {
            "full" => Self::Full,
            _ => Self::AppendOnly,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelSnapshotOutput {
    pub observation_v1: Vec<f32>,
    pub observation_v5: Vec<f32>,
    pub action_mask: Vec<f32>,
    pub action_features: Vec<f32>,
    /// Parallel binary mana_draw head — true when mana_draw is a legal
    /// action this turn for the snapshotted player (Phase 2: MD-3). This is
    /// NOT part of the 601 action_mask; it is exposed as a separate per-step
    /// flag so the model/policy can choose mana_draw via a parallel channel.
    pub mana_draw_legal: bool,
}

#[derive(Debug, Clone)]
pub struct KernelStepOutput {
    pub state: KernelState,
    pub reward: f32,
    pub terminated: bool,
    pub reward_components_v5: RewardComponentsV5,
}

impl KernelSnapshotOutput {
    pub fn legal_ids(&self) -> Vec<usize> {
        self.action_mask
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if *v == 1.0 { Some(i) } else { None })
            .collect()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RolloutKernel {
    config: KernelConfig,
}

impl RolloutKernel {
    pub fn new(config: KernelConfig) -> Self {
        Self { config }
    }

    pub fn encode_snapshot(&self, state: &KernelState, player_id: i32) -> KernelSnapshotOutput {
        self.encode_snapshot_with_history(state, player_id, &[])
    }

    pub fn encode_snapshot_with_history(
        &self,
        state: &KernelState,
        player_id: i32,
        history_events: &[KernelHistoryEvent],
    ) -> KernelSnapshotOutput {
        let action_mask = build_action_mask(state, player_id, self.config.placement_mode);
        let action_features = encode_action_features(state, player_id, &action_mask);
        let observation_v1 = encode_observation_v1(state, player_id);
        let observation_v5 = encode_observation_v5_from_v1(
            state,
            player_id,
            self.config.info_mode,
            self.config.assist_mode,
            history_events,
            &observation_v1,
        );
        let (me, _enemy) = state.players_for(player_id);
        let mana_draw_legal = mana_draw_legal_for(me);
        KernelSnapshotOutput {
            observation_v1,
            observation_v5,
            action_mask,
            action_features,
            mana_draw_legal,
        }
    }

    pub fn legal_action_ids(&self, state: &KernelState, player_id: i32) -> Vec<usize> {
        build_action_mask(state, player_id, self.config.placement_mode)
            .iter()
            .enumerate()
            .filter_map(|(action_id, value)| if *value == 1.0 { Some(action_id) } else { None })
            .collect()
    }

    pub fn apply_action(
        &self,
        state: &KernelState,
        player_id: i32,
        action_id: usize,
        mana_draw_flag: bool,
        draw_rng: &mut DrawRng,
    ) -> Result<KernelStepOutput, String> {
        if state.status != "ongoing" {
            return Err("game_over".to_string());
        }
        if state.current_turn_owner_id != player_id {
            return Err("not_your_turn".to_string());
        }

        // Parallel binary mana_draw head (Phase 2: MD-3, AC-FFI-1/MD-FFI).
        // mana_draw is a STANDALONE action type that REPLACES the action_id
        // decode — it is NOT a 601st candidate. When `mana_draw_flag` is set,
        // the env applies a mana-draw for the current player instead of
        // decoding `action_id`. Legality is verified via `mana_draw_legal_for`
        // (separate from the 601 action_mask). The 601 codec layout is frozen.
        if mana_draw_flag {
            let (actor, _opponent) = state.players_for(player_id);
            if !mana_draw_legal_for(actor) {
                return Err("illegal_action".to_string());
            }
            let mut next = state.clone();
            let overdraw_to_discard = self.config.overdraw_to_discard;
            let (player, _opponent) = next.players_for_mut(player_id)?;
            apply_mana_draw(player, overdraw_to_discard, draw_rng)?;
            cleanup_dead_units(&mut next);
            check_game_over(&mut next);
            let base_reward = compute_trainv2_reward(state, &next, player_id);
            let reward_components_v5 = compute_reward_components_v5(state, &next, player_id);
            let reward = if self.config.v5_weighted_reward {
                compute_weighted_reward_v5(base_reward, reward_components_v5, self.config.info_mode)
            } else {
                base_reward
            };
            let terminated = next.status != "ongoing";
            return Ok(KernelStepOutput {
                reward_components_v5,
                reward,
                terminated,
                state: next,
            });
        }

        let mask = build_action_mask(state, player_id, self.config.placement_mode);
        if action_id >= mask.len() || mask[action_id] != 1.0 {
            return Err("illegal_action".to_string());
        }

        let mut next = state.clone();
        let overdraw_to_discard = self.config.overdraw_to_discard;
        match decode_action_id(action_id) {
            Some(CandidateAction::EndTurn) => apply_end_turn(
                &mut next,
                player_id,
                self.config.mana_per_turn,
                overdraw_to_discard,
                draw_rng,
            )?,
            Some(CandidateAction::PlayCard {
                hand_index,
                board_position,
                target_code,
            }) => apply_play_card(
                &mut next,
                player_id,
                hand_index,
                board_position,
                target_code,
                overdraw_to_discard,
                draw_rng,
            )?,
            Some(CandidateAction::Attack {
                attacker_index,
                target_code,
            }) => apply_attack(&mut next, player_id, attacker_index, target_code)?,
            None => return Err("unknown_action".to_string()),
        }

        cleanup_dead_units(&mut next);
        check_game_over(&mut next);

        let base_reward = compute_trainv2_reward(state, &next, player_id);
        let reward_components_v5 = compute_reward_components_v5(state, &next, player_id);
        let reward = if self.config.v5_weighted_reward {
            compute_weighted_reward_v5(base_reward, reward_components_v5, self.config.info_mode)
        } else {
            base_reward
        };
        let terminated = next.status != "ongoing";

        Ok(KernelStepOutput {
            reward_components_v5,
            reward,
            terminated,
            state: next,
        })
    }
}

pub fn hash_f32_le(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

pub fn build_action_mask(
    state: &KernelState,
    player_id: i32,
    placement_mode: PlacementMode,
) -> Vec<f32> {
    let mut mask = vec![0.0_f32; MAX_CANDIDATE_ACTIONS];
    if state.status != "ongoing" || state.current_turn_owner_id != player_id {
        return mask;
    }

    let (me, enemy) = state.players_for(player_id);
    mask[0] = 1.0;
    mask_play_actions(&mut mask, me, enemy);
    mask_attack_actions(&mut mask, me, enemy);
    apply_placement_mode(&mut mask, me, placement_mode);
    mask
}

fn mask_play_actions(mask: &mut [f32], me: &KernelPlayer, enemy: &KernelPlayer) {
    for hand_idx in 0..me.hand.len().min(NUM_HAND) {
        let card = &me.hand[hand_idx];
        if me.mana < card.mana_cost {
            continue;
        }
        let is_warrior = card.is_warrior();
        if is_warrior && me.board.len() >= GAME_BOARD_CAP {
            continue;
        }

        let needs_target =
            requires_target(&card.mechanics) && !is_random_battlecry_damage_card(card);
        let has_choose_shield_damage = card.has_mechanic("choose_shield_damage");
        let num_positions = if is_warrior {
            (me.board.len() + 1).min(NUM_PLAY_POS).max(1)
        } else {
            1
        };

        for pos_idx in 0..num_positions {
            let base = PLAY_BASE + hand_idx * PLAY_STRIDE + pos_idx * NUM_PLAY_TARGETS;
            if !needs_target && !has_choose_shield_damage {
                mask[base] = 1.0;
                continue;
            }
            mask_targets_for_card(mask, base, card, me, enemy);
        }
    }
}

fn mask_attack_actions(mask: &mut [f32], me: &KernelPlayer, enemy: &KernelPlayer) {
    let taunt_indices: Vec<usize> = enemy
        .board
        .iter()
        .enumerate()
        .filter_map(|(i, unit)| {
            if unit.has_mechanic("taunt") {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let has_taunts = !taunt_indices.is_empty();

    for att_idx in 0..me.board.len().min(NUM_BOARD) {
        let attacker = &me.board[att_idx];
        if !attacker.is_ready || attacker.is_frozen {
            continue;
        }
        let eff_atk = compute_effective_attack(attacker, &me.board, &me.hero);
        if eff_atk <= 0 {
            continue;
        }
        let can_bypass = attacker.has_mechanic("bypass_taunt");
        let base = ATTACK_BASE + att_idx * NUM_ATTACK_TARGETS;
        for target_idx in 0..enemy.board.len() {
            if has_taunts && !can_bypass {
                if taunt_indices.contains(&target_idx) {
                    mask[base + target_idx] = 1.0;
                }
            } else {
                mask[base + target_idx] = 1.0;
            }
        }
        if !has_taunts || can_bypass {
            mask[base + 7] = 1.0;
        }
    }
}

fn mask_targets_for_card(
    mask: &mut [f32],
    base: usize,
    card: &KernelCard,
    me: &KernelPlayer,
    enemy: &KernelPlayer,
) {
    let mechanics = &card.mechanics;
    let is_consume = mechanics.iter().any(|m| m.contains("consume_ally"));
    let is_damage = mechanics.iter().any(|m| m.contains("damage"));
    let is_heal_target = mechanics.iter().any(|m| m.contains("heal_target"));
    let is_heal = mechanics.iter().any(|m| m.contains("heal"));
    let is_buff = mechanics.iter().any(|m| m.contains("buff"));
    let is_delete = mechanics.iter().any(|m| m.contains("delete_target"));
    let is_freeze = mechanics
        .iter()
        .any(|m| m.contains("freeze") || m.contains("battlecry_freeze"));
    let is_choose_shield_damage = mechanics.iter().any(|m| m.contains("choose_shield_damage"));

    if is_consume {
        for target_idx in 0..me.board.len() {
            mask[base + 9 + target_idx] = 1.0;
        }
        return;
    }

    if is_choose_shield_damage {
        for target_idx in 0..enemy.board.len() {
            mask[base + 1 + target_idx] = 1.0;
        }
        mask[base + 8] = 1.0;
        mask[base] = 1.0;
        return;
    }

    if is_freeze && !is_damage {
        for target_idx in 0..enemy.board.len() {
            mask[base + 1 + target_idx] = 1.0;
        }
        return;
    }

    if is_damage || is_freeze {
        for target_idx in 0..enemy.board.len() {
            mask[base + 1 + target_idx] = 1.0;
        }
        mask[base + 8] = 1.0;
        return;
    }

    if is_delete {
        for target_idx in 0..enemy.board.len() {
            mask[base + 1 + target_idx] = 1.0;
        }
        return;
    }

    if is_heal_target {
        for target_idx in 0..me.board.len() {
            mask[base + 9 + target_idx] = 1.0;
        }
        mask[base + 16] = 1.0;
        return;
    }

    if is_heal {
        for (target_idx, unit) in me.board.iter().enumerate() {
            if unit.hp < unit.max_hp {
                mask[base + 9 + target_idx] = 1.0;
            }
        }
        if me.hero.hp < me.hero.max_hp {
            mask[base + 16] = 1.0;
        }
        return;
    }

    if is_buff {
        for target_idx in 0..me.board.len() {
            mask[base + 9 + target_idx] = 1.0;
        }
    }
}

fn apply_placement_mode(mask: &mut [f32], me: &KernelPlayer, placement_mode: PlacementMode) {
    if placement_mode == PlacementMode::Full {
        return;
    }
    let expected_pos = me.board.len();
    for hand_idx in 0..me.hand.len().min(NUM_HAND) {
        let card = &me.hand[hand_idx];
        if !card.is_warrior() {
            continue;
        }
        for pos_idx in 0..NUM_PLAY_POS {
            if pos_idx == expected_pos {
                continue;
            }
            let base = PLAY_BASE + hand_idx * PLAY_STRIDE + pos_idx * NUM_PLAY_TARGETS;
            for target_code in 0..NUM_PLAY_TARGETS {
                mask[base + target_code] = 0.0;
            }
        }
    }
}

/// Whether mana_draw is a legal action for `player` this turn — mirrors
/// `core/engine.py` legal-actions generation (hand not full AND mana covers
/// the next cost step `MANA_DRAW_BASE * (count + 1)`). Phase 2: MD-3.
pub fn mana_draw_legal_for(player: &KernelPlayer) -> bool {
    if player.hand.len() >= HAND_CAP {
        return false;
    }
    let cost = MANA_DRAW_BASE * (player.mana_draw_count_this_turn + 1);
    player.mana >= cost
}

/// Apply a mana-draw action for `player` — mirrors
/// `core/engine.py::_handle_mana_draw` EXACTLY (Phase 2: MD-1/2/5):
/// - guard hand cap (HAND_CAP=4) → "hand_full"
/// - cost = MANA_DRAW_BASE * (count + 1); guard mana → "insufficient_mana"
/// - deduct mana BEFORE the draw (MD-5)
/// - on draw failure (fatigue: empty deck + graveyard) REFUND the mana and
///   do NOT increment the counter → "no_cards_to_draw" (matches Python; the
///   Python code explicitly refunds `player.mana += cost`)
/// - on success, `mana_draw_count_this_turn += 1`
pub fn apply_mana_draw(
    player: &mut KernelPlayer,
    overdraw_to_discard: bool,
    draw_rng: &mut DrawRng,
) -> Result<(), String> {
    if player.hand.len() >= HAND_CAP {
        return Err("hand_full".to_string());
    }
    let cost = MANA_DRAW_BASE * (player.mana_draw_count_this_turn + 1);
    if player.mana < cost {
        return Err("insufficient_mana".to_string());
    }
    player.mana -= cost;
    let drawn_ok = draw_one_from_deck(player, overdraw_to_discard, draw_rng);
    if !drawn_ok {
        // Fatigue: deck + graveyard empty — refund mana, counter unchanged
        // (mirrors Python `_handle_mana_draw` `player.mana += cost`).
        player.mana += cost;
        return Err("no_cards_to_draw".to_string());
    }
    player.mana_draw_count_this_turn += 1;
    Ok(())
}

fn apply_end_turn(
    state: &mut KernelState,
    player_id: i32,
    mana_per_turn: i32,
    overdraw_to_discard: bool,
    draw_rng: &mut DrawRng,
) -> Result<(), String> {
    let next_player_id = {
        let (_player, opponent) = state.players_for(player_id);
        opponent.user_id
    };
    state.current_turn_owner_id = next_player_id;
    state.turn_number += 1;

    let opponent = state.player_mut(next_player_id)?;
    opponent.max_mana = (opponent.max_mana + mana_per_turn).min(10);
    opponent.mana = opponent.max_mana;
    // mana_draw_count_this_turn resets at the start of each player's turn
    // (mirrors core/engine.py _handle_end_turn).
    opponent.mana_draw_count_this_turn = 0;

    for unit in &mut opponent.board {
        if unit.is_frozen {
            unit.is_ready = false;
            unit.is_frozen = false;
        } else {
            unit.is_ready = true;
        }
        if unit.card_id == 24 && !unit.has_mechanic("shield") {
            unit.mechanics.push("shield".to_string());
        }
        apply_regen(unit);
    }
    apply_regen(&mut opponent.hero);

    // No-FIFO weighted draw — mirrors core/engine.py draw_one_from_deck
    // (source="end_turn"). Replaces the old `deck.remove(0)` FIFO draw.
    draw_one_from_deck(opponent, overdraw_to_discard, draw_rng);
    Ok(())
}

fn apply_play_card(
    state: &mut KernelState,
    player_id: i32,
    hand_index: usize,
    board_position: usize,
    target_code: usize,
    overdraw_to_discard: bool,
    draw_rng: &mut DrawRng,
) -> Result<(), String> {
    let (player, opponent) = state.players_for_mut(player_id)?;
    if hand_index >= player.hand.len() {
        return Err("invalid_hand_index".to_string());
    }

    let card = &player.hand[hand_index];
    if player.mana < card.mana_cost {
        return Err("insufficient_mana".to_string());
    }
    if card.is_warrior() && player.board.len() >= GAME_BOARD_CAP && !card.has_mechanic("consume_ally") {
        return Err("board_full".to_string());
    }
    if requires_target(&card.mechanics)
        && target_code == 0
        && !is_random_battlecry_damage_card(card)
    {
        return Err("target_required".to_string());
    }

    player.mana -= card.mana_cost;
    let mut card = player.hand.remove(hand_index);
    if card.is_warrior() {
        let position = board_position.min(player.board.len());
        card.is_ready = card.has_mechanic("charge");
        let card_id = card.card_id;
        let mechanics = card.mechanics.clone();
        player.board.insert(position, card);
        apply_play_effects(
            card_id,
            &mechanics,
            player,
            opponent,
            target_code,
            overdraw_to_discard,
            draw_rng,
        );
    } else if card.is_potion() {
        let card_id = card.card_id;
        let mechanics = card.mechanics.clone();
        apply_play_effects(
            card_id,
            &mechanics,
            player,
            opponent,
            target_code,
            overdraw_to_discard,
            draw_rng,
        );
        player.graveyard.push(card);
    }
    Ok(())
}

fn apply_attack(
    state: &mut KernelState,
    player_id: i32,
    attacker_index: usize,
    target_code: usize,
) -> Result<(), String> {
    let (player, opponent) = state.players_for_mut(player_id)?;
    if attacker_index >= player.board.len() {
        return Err("attacker_not_found".to_string());
    }
    if !player.board[attacker_index].is_ready {
        return Err("unit_not_ready".to_string());
    }

    let effective_attack =
        compute_effective_attack(&player.board[attacker_index], &player.board, &player.hero);
    if effective_attack <= 0 {
        return Err("no_attack".to_string());
    }

    if target_code == 7 {
        let has_lifesteal = player.board[attacker_index].has_mechanic("lifesteal");
        let damage_dealt = {
            let attacker = &mut player.board[attacker_index];
            let damage_dealt = apply_damage(&mut opponent.hero, effective_attack, Some(attacker));
            attacker.is_ready = false;
            damage_dealt
        };
        if has_lifesteal && damage_dealt > 0 {
            heal_card(&mut player.hero, damage_dealt);
        }
        return Ok(());
    }

    if target_code >= opponent.board.len() {
        return Err("target_not_found".to_string());
    }

    let target_effective_attack = compute_effective_attack(
        &opponent.board[target_code],
        &opponent.board,
        &opponent.hero,
    );
    let attacker_effective_attack =
        compute_effective_attack(&player.board[attacker_index], &player.board, &player.hero);
    let has_lifesteal = player.board[attacker_index].has_mechanic("lifesteal");
    let damage_dealt_to_target = {
        let attacker = &mut player.board[attacker_index];
        apply_damage(
            &mut opponent.board[target_code],
            attacker_effective_attack,
            Some(attacker),
        )
    };
    {
        let target = &mut opponent.board[target_code];
        apply_damage(
            &mut player.board[attacker_index],
            target_effective_attack,
            Some(target),
        );
    }
    player.board[attacker_index].is_ready = false;
    if has_lifesteal && damage_dealt_to_target > 0 {
        heal_card(&mut player.hero, damage_dealt_to_target);
    }
    Ok(())
}

fn apply_play_effects(
    card_id: i32,
    mechanics: &[String],
    owner: &mut KernelPlayer,
    opponent: &mut KernelPlayer,
    target_code: usize,
    overdraw_to_discard: bool,
    draw_rng: &mut DrawRng,
) {
    for mechanic in mechanics {
        if mechanic.starts_with("deathrattle_") {
            continue;
        }
        if let Some(amount) = mechanic
            .strip_prefix("battlecry_damage_")
            .and_then(parse_i32_prefix)
        {
            if card_id != 15 {
                apply_damage_to_play_target(owner, opponent, target_code, amount);
            }
        } else if let Some(amount) = mechanic.strip_prefix("damage_").and_then(parse_i32_prefix) {
            apply_damage_to_play_target(owner, opponent, target_code, amount);
        } else if let Some(amount) = mechanic
            .strip_prefix("battlecry_heal_hero_")
            .and_then(parse_i32_prefix)
        {
            owner.hero.hp = (owner.hero.hp + amount).min(owner.hero.max_hp);
        } else if mechanic == "battlecry_draw_card" {
            // No-FIFO weighted draw — mirrors core/engine.py draw_one_from_deck
            // (source="battlecry_draw_card"). Replaces the old FIFO draw.
            draw_one_from_deck(owner, overdraw_to_discard, draw_rng);
        }
    }
}

fn apply_damage_to_play_target(
    owner: &mut KernelPlayer,
    opponent: &mut KernelPlayer,
    target_code: usize,
    amount: i32,
) {
    match target_code {
        1..=7 => {
            if let Some(target) = opponent.board.get_mut(target_code - 1) {
                apply_damage(target, amount, None);
            }
        }
        8 => {
            apply_damage(&mut opponent.hero, amount, None);
        }
        16 => {
            apply_damage(&mut owner.hero, amount, None);
        }
        9..=15 => {
            if let Some(target) = owner.board.get_mut(target_code - 9) {
                apply_damage(target, amount, None);
            }
        }
        _ => {}
    }
}

pub fn encode_action_features(state: &KernelState, player_id: i32, mask: &[f32]) -> Vec<f32> {
    let mut features = vec![0.0_f32; MAX_CANDIDATE_ACTIONS * ACTION_FEATURE_DIM_V1];
    let (me, enemy) = state.players_for(player_id);
    for action_id in 0..MAX_CANDIDATE_ACTIONS {
        if mask[action_id] != 1.0 {
            continue;
        }
        let start = action_id * ACTION_FEATURE_DIM_V1;
        encode_one_action(
            &mut features[start..start + ACTION_FEATURE_DIM_V1],
            me,
            enemy,
            action_id,
        );
    }
    features
}

fn encode_one_action(out: &mut [f32], me: &KernelPlayer, enemy: &KernelPlayer, action_id: usize) {
    let source_card: Option<&KernelCard>;
    let target_card: Option<&KernelCard>;
    let mut board_pos: Option<usize> = None;
    let mut hand_pos: Option<usize> = None;
    let mut attacker_pos: Option<usize> = None;
    let rel: [f32; 8];

    match decode_action_id(action_id) {
        Some(CandidateAction::EndTurn) => {
            out[128] = 1.0;
            return;
        }
        Some(CandidateAction::PlayCard {
            hand_index,
            board_position,
            target_code,
        }) => {
            out[129] = 1.0;
            source_card = me.hand.get(hand_index);
            hand_pos = Some(hand_index);
            if let Some(card) = source_card {
                if card.is_warrior() {
                    board_pos = Some(board_position);
                }
            }
            let target = resolve_target_card(me, enemy, target_code, false);
            target_card = target.0;
            rel = target.1;
        }
        Some(CandidateAction::Attack {
            attacker_index,
            target_code,
        }) => {
            out[130] = 1.0;
            source_card = me.board.get(attacker_index);
            attacker_pos = Some(attacker_index);
            let target = resolve_target_card(me, enemy, target_code, true);
            target_card = target.0;
            rel = target.1;
        }
        None => return,
    }

    let effective_attack = source_card.and_then(|card| {
        if card.is_warrior() {
            Some(compute_effective_attack(card, &me.board, &me.hero))
        } else {
            None
        }
    });
    let source_shape = source_card
        .map(|card| card.shape(board_pos, hand_pos, effective_attack))
        .unwrap_or([0.0; CARD_SHAPE_DIM]);
    let target_shape = target_card
        .map(|card| card.shape(None, None, None))
        .unwrap_or([0.0; CARD_SHAPE_DIM]);

    out[..CARD_SHAPE_DIM].copy_from_slice(&source_shape);
    out[CARD_SHAPE_DIM..2 * CARD_SHAPE_DIM].copy_from_slice(&target_shape);
    out[131..139].copy_from_slice(&rel);
    out[139] = board_pos
        .map(|p| ((p as f64 + 1.0) / 9.0) as f32)
        .unwrap_or(0.0);
    out[140] = hand_pos
        .map(|p| ((p as f64 + 1.0) / 5.0) as f32)
        .unwrap_or(0.0);
    out[141] = attacker_pos
        .map(|p| ((p as f64 + 1.0) / 8.0) as f32)
        .unwrap_or(0.0);
}

fn resolve_target_card<'a>(
    me: &'a KernelPlayer,
    enemy: &'a KernelPlayer,
    target_code: usize,
    is_attack: bool,
) -> (Option<&'a KernelCard>, [f32; 8]) {
    let mut rel = [0.0_f32; 8];
    if is_attack {
        if target_code <= 6 {
            if let Some(card) = enemy.board.get(target_code) {
                rel[1] = 1.0;
                return (Some(card), rel);
            }
        }
        if target_code == 7 {
            rel[0] = 1.0;
            return (Some(&enemy.hero), rel);
        }
        return (None, rel);
    }

    if target_code == 0 {
        rel[7] = 1.0;
        return (None, rel);
    }
    if target_code == 8 {
        rel[0] = 1.0;
        return (Some(&enemy.hero), rel);
    }
    if target_code == 16 {
        rel[2] = 1.0;
        return (Some(&me.hero), rel);
    }
    if (1..=7).contains(&target_code) {
        if let Some(card) = enemy.board.get(target_code - 1) {
            rel[1] = 1.0;
            return (Some(card), rel);
        }
    }
    if (9..=15).contains(&target_code) {
        if let Some(card) = me.board.get(target_code - 9) {
            rel[3] = 1.0;
            return (Some(card), rel);
        }
    }
    (None, rel)
}

pub fn encode_observation_v1(state: &KernelState, player_id: i32) -> Vec<f32> {
    let mut out = vec![0.0_f32; OBS_DIM_V1];
    let (me, enemy) = state.players_for(player_id);
    encode_globals_v1(&mut out, state, me, enemy);
    let mut offset = GLOBAL_DIM_V1;
    encode_card_slots_v1(&mut out, &mut offset, me, enemy);
    encode_zone_summaries_v1(&mut out, offset, me, enemy);
    out
}

fn encode_globals_v1(
    out: &mut [f32],
    state: &KernelState,
    me: &KernelPlayer,
    enemy: &KernelPlayer,
) {
    let status_idx = match state.status.as_str() {
        "ongoing" => 0,
        "p1_win" => 1,
        "p2_win" => 2,
        _ => 3,
    };
    out[0] = norm(state.turn_number as f32, 50.0);
    out[1] = if state.current_turn_owner_id == me.user_id {
        1.0
    } else {
        0.0
    };
    out[2] = if status_idx == 0 { 1.0 } else { 0.0 };
    out[3] = if status_idx == 1 { 1.0 } else { 0.0 };
    out[4] = if status_idx == 2 { 1.0 } else { 0.0 };
    out[5] = if status_idx == 3 { 1.0 } else { 0.0 };
    out[6] = norm(me.mana as f32, 10.0);
    out[7] = norm(me.max_mana as f32, 10.0);
    out[8] = norm(me.hand.len() as f32, 4.0);
    out[9] = norm(me.board.len() as f32, 7.0);
    out[10] = norm(me.deck.len() as f32, 12.0);
    out[11] = norm(me.graveyard.len() as f32, 16.0);
    out[12] = norm(me.hero.hp as f32, 50.0);
    out[13] = norm(me.hero.max_hp as f32, 50.0);
    out[14] = norm(me.board.iter().map(|c| c.attack).sum::<i32>() as f32, 50.0);
    out[15] = norm(me.board.iter().map(|c| c.hp).sum::<i32>() as f32, 100.0);
    out[16] = norm(me.board.iter().filter(|c| c.is_ready).count() as f32, 7.0);
    out[17] = norm(enemy.mana as f32, 10.0);
    out[18] = norm(enemy.max_mana as f32, 10.0);
    out[19] = norm(enemy.hand.len() as f32, 4.0);
    out[20] = norm(enemy.board.len() as f32, 7.0);
    out[21] = norm(enemy.deck.len() as f32, 12.0);
    out[22] = norm(enemy.graveyard.len() as f32, 16.0);
    out[23] = norm(enemy.hero.hp as f32, 50.0);
    out[24] = norm(enemy.hero.max_hp as f32, 50.0);
    out[25] = norm(
        enemy.board.iter().map(|c| c.attack).sum::<i32>() as f32,
        50.0,
    );
    out[26] = norm(enemy.board.iter().map(|c| c.hp).sum::<i32>() as f32, 100.0);
    out[27] = norm(
        enemy.board.iter().filter(|c| c.is_ready).count() as f32,
        7.0,
    );
    out[28] = norm(me.trophies as f32, 5000.0);
    out[29] = norm(enemy.trophies as f32, 5000.0);
}

fn encode_card_slots_v1(
    out: &mut [f32],
    offset: &mut usize,
    me: &KernelPlayer,
    enemy: &KernelPlayer,
) {
    write_shape(out, offset, Some(&me.hero), None, None, None);
    write_shape(out, offset, Some(&enemy.hero), None, None, None);
    for i in 0..7 {
        write_shape(out, offset, me.board.get(i), Some(i), None, None);
    }
    for i in 0..7 {
        write_shape(out, offset, enemy.board.get(i), Some(i), None, None);
    }
    for i in 0..4 {
        write_shape(out, offset, me.hand.get(i), None, Some(i), None);
    }
}

fn write_shape(
    out: &mut [f32],
    offset: &mut usize,
    card: Option<&KernelCard>,
    board_pos: Option<usize>,
    hand_pos: Option<usize>,
    effective_attack: Option<i32>,
) {
    let shape = card
        .map(|card| card.shape(board_pos, hand_pos, effective_attack))
        .unwrap_or([0.0; CARD_SHAPE_DIM]);
    out[*offset..*offset + CARD_SHAPE_DIM].copy_from_slice(&shape);
    *offset += CARD_SHAPE_DIM;
}

fn encode_zone_summaries_v1(
    out: &mut [f32],
    offset: usize,
    me: &KernelPlayer,
    enemy: &KernelPlayer,
) {
    encode_one_zone_v1(out, offset, &me.deck);
    encode_one_zone_v1(out, offset + ZONE_SUMMARY_DIM_V1, &me.graveyard);
    encode_one_zone_v1(out, offset + 2 * ZONE_SUMMARY_DIM_V1, &enemy.graveyard);
}

fn encode_one_zone_v1(out: &mut [f32], base: usize, cards: &[KernelCard]) {
    let n = cards.len();
    if n == 0 {
        return;
    }
    let n_f = n as f32;
    let mana_sum = cards.iter().map(|c| c.mana_cost).sum::<i32>();
    let attack_sum = cards.iter().map(|c| c.attack).sum::<i32>();
    let hp_sum = cards.iter().map(|c| c.hp).sum::<i32>();
    out[base] = norm(n_f, 12.0);
    out[base + 1] = norm64(mana_sum as f64 / n as f64, 10.0);
    out[base + 2] = norm64(attack_sum as f64 / n as f64, 20.0);
    out[base + 3] = norm64(hp_sum as f64 / n as f64, 20.0);
    out[base + 4] = norm(
        cards.iter().map(|c| c.mana_cost).max().unwrap_or(0) as f32,
        10.0,
    );
    out[base + 5] = norm(
        cards.iter().map(|c| c.attack).max().unwrap_or(0) as f32,
        20.0,
    );
    out[base + 6] = norm(cards.iter().map(|c| c.hp).max().unwrap_or(0) as f32, 20.0);
    out[base + 7] = norm(attack_sum as f32, 50.0);
    out[base + 8] = cards.iter().filter(|c| c.is_warrior()).count() as f32 / n_f;
    out[base + 9] = cards.iter().filter(|c| c.is_potion()).count() as f32 / n_f;
    out[base + 10] = norm64(
        cards.iter().map(|c| c.level).sum::<i32>() as f64 / n as f64,
        10.0,
    );

    for (mechanic_idx, mechanic_name) in MECHANICS_LIST.iter().enumerate() {
        let count = cards
            .iter()
            .filter(|card| {
                card.mechanics
                    .iter()
                    .any(|m| m == mechanic_name || m.starts_with(&format!("{mechanic_name}_")))
            })
            .count();
        out[base + 11 + mechanic_idx] = count as f32 / n_f;
    }
    out[base + 44] = 0.0;
    out[base + 45] = 0.0;
    out[base + 46] = 0.0;
    out[base + 47] = 0.0;
}

pub fn encode_observation_v5(
    state: &KernelState,
    player_id: i32,
    info_mode: InfoModeV5,
    assist_mode: AssistModeV5,
    history_events: &[KernelHistoryEvent],
) -> Vec<f32> {
    let obs_v1 = encode_observation_v1(state, player_id);
    encode_observation_v5_from_v1(
        state,
        player_id,
        info_mode,
        assist_mode,
        history_events,
        &obs_v1,
    )
}

fn encode_observation_v5_from_v1(
    state: &KernelState,
    player_id: i32,
    info_mode: InfoModeV5,
    assist_mode: AssistModeV5,
    history_events: &[KernelHistoryEvent],
    observation_v1: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(observation_v1.len(), OBS_DIM_V1);
    let mut out = vec![0.0_f32; OBS_DIM_V5];
    out[..OBS_DIM_V1].copy_from_slice(observation_v1);
    let global_base = OBS_DIM_V1;
    out[global_base] = info_mode.clipped_strength();
    out[global_base + 1] = bool_f32(info_mode.own_hand_identity_known);
    out[global_base + 2] = bool_f32(info_mode.own_deck_known);
    out[global_base + 3] = bool_f32(info_mode.enemy_hand_known);
    out[global_base + 4] = bool_f32(info_mode.enemy_deck_known);
    out[global_base + 5] = bool_f32(info_mode.enemy_deck_order_known);
    out[global_base + 6] = (history_events.len() as f32 / HISTORY_EVENTS as f32).min(1.0);
    out[global_base + 7] = bool_f32(info_mode.draw_assist_enabled);
    out[global_base + 8] = info_mode.clipped_draw_assist_strength();
    out[global_base + 9] = bool_f32(assist_mode.assembler_enabled);
    out[global_base + 10] = assist_mode.clipped_assembler_strength();
    out[global_base + 11] = bool_f32(assist_mode.desirerer_enabled);
    out[global_base + 12] = assist_mode.clipped_desirerer_strength();
    out[global_base + 13] = bool_f32(assist_mode.teacher_hint_available);
    out[global_base + 14] = assist_mode.clipped_profile_id() as f32 / 16.0;

    let private_base = global_base + V5_GLOBAL_DIM;
    encode_private_info_v5(
        &mut out[private_base..private_base + PRIVATE_INFO_DIM],
        state,
        player_id,
        info_mode,
    );
    let history_base = private_base + PRIVATE_INFO_DIM;
    debug_assert_eq!(out[history_base..].len(), HISTORY_DIM);
    encode_history_v5(&mut out[history_base..], player_id, history_events);
    out
}

fn encode_private_info_v5(
    out: &mut [f32],
    state: &KernelState,
    player_id: i32,
    info_mode: InfoModeV5,
) {
    let (me, enemy) = state.players_for(player_id);
    let mut offset = 0;
    offset = encode_zone_v5(
        out,
        offset,
        &me.hand,
        OWN_HAND_SLOTS,
        info_mode.own_hand_identity_known,
    );
    offset = encode_zone_v5(
        out,
        offset,
        &me.deck,
        OWN_DECK_SLOTS,
        info_mode.own_deck_known,
    );
    offset = encode_zone_v5(
        out,
        offset,
        &enemy.hand,
        ENEMY_HAND_SLOTS,
        info_mode.enemy_hand_known,
    );
    encode_zone_v5(
        out,
        offset,
        &enemy.deck,
        ENEMY_DECK_SLOTS,
        info_mode.enemy_deck_known,
    );
}

fn encode_zone_v5(
    out: &mut [f32],
    offset: usize,
    cards: &[KernelCard],
    slots: usize,
    known: bool,
) -> usize {
    for slot in 0..slots {
        let base = offset + slot * PRIVATE_CARD_SLOT_DIM;
        if known {
            if let Some(card) = cards.get(slot) {
                out[base] = 1.0;
                out[base + 1] = norm(card.card_id as f32, CARD_ID_NORMALIZER);
                let shape = card.shape(None, None, None);
                out[base + 2..base + 2 + CARD_SHAPE_DIM].copy_from_slice(&shape);
            }
        }
    }
    offset + slots * PRIVATE_CARD_SLOT_DIM
}

fn encode_history_v5(out: &mut [f32], player_id: i32, events: &[KernelHistoryEvent]) {
    let recent_start = events.len().saturating_sub(HISTORY_EVENTS);
    let recent = &events[recent_start..];
    let start_slot = HISTORY_EVENTS - recent.len();
    for (idx, event) in recent.iter().enumerate() {
        let base = (start_slot + idx) * HISTORY_EVENT_DIM;
        encode_one_history_event_v5(&mut out[base..base + HISTORY_EVENT_DIM], player_id, event);
    }
}

fn encode_one_history_event_v5(out: &mut [f32], player_id: i32, event: &KernelHistoryEvent) {
    out[0] = 1.0;
    out[1] = bool_f32(event.actor_id == player_id);
    out[2] = bool_f32(event.actor_id != 0 && event.actor_id != player_id);
    out[3] = bool_f32(event.action_type == "end_turn");
    out[4] = bool_f32(event.action_type == "play_card");
    out[5] = bool_f32(event.action_type == "attack");
    out[6] = norm(event.action_id as f32, 600.0);
    out[7] = signed_norm(event.enemy_hero_hp_delta as f32, 50.0);
    out[8] = signed_norm(event.own_hero_hp_delta as f32, 50.0);
    out[9] = signed_norm(event.my_board_count_delta as f32, 7.0);
    out[10] = signed_norm(event.enemy_board_count_delta as f32, 7.0);
    out[11] = norm(event.turn_number as f32, 50.0);
    out[12] = signed_norm(event.board_power_delta, 200.0);
    if let Some(card) = &event.source_card {
        let shape = card.shape(None, None, None);
        out[16..16 + CARD_SHAPE_DIM].copy_from_slice(&shape);
    }
    if let Some(card) = &event.target_card {
        let shape = card.shape(None, None, None);
        out[80..80 + CARD_SHAPE_DIM].copy_from_slice(&shape);
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct RewardComponentsV5 {
    pub hp_potential_delta: f32,
    pub board_power_delta: f32,
    pub my_board_power: f32,
    pub enemy_board_power: f32,
    pub board_power_ratio: f32,
    pub board_under_0_7: bool,
    pub own_board_wiped: bool,
    pub my_board_count_delta: i32,
    pub enemy_board_count_delta: i32,
}

pub fn compute_reward_components_v5(
    pre: &KernelState,
    post: &KernelState,
    player_id: i32,
) -> RewardComponentsV5 {
    let pre = RewardSnapshotV5::from_state(pre, player_id);
    let post = RewardSnapshotV5::from_state(post, player_id);
    let pre_hp_potential = missing_ratio(pre.enemy_hero_hp, pre.enemy_hero_max_hp)
        - missing_ratio(pre.my_hero_hp, pre.my_hero_max_hp);
    let post_hp_potential = missing_ratio(post.enemy_hero_hp, post.enemy_hero_max_hp)
        - missing_ratio(post.my_hero_hp, post.my_hero_max_hp);
    let pre_board_delta = pre.my_board_power - pre.enemy_board_power;
    let post_board_delta = post.my_board_power - post.enemy_board_power;
    let ratio = post.my_board_power / post.enemy_board_power.max(1.0);
    RewardComponentsV5 {
        hp_potential_delta: post_hp_potential - pre_hp_potential,
        board_power_delta: post_board_delta - pre_board_delta,
        my_board_power: post.my_board_power,
        enemy_board_power: post.enemy_board_power,
        board_power_ratio: ratio,
        board_under_0_7: post.enemy_board_power > 0.0 && ratio < 0.7,
        own_board_wiped: pre.my_board_count > 0 && post.my_board_count == 0,
        my_board_count_delta: post.my_board_count - pre.my_board_count,
        enemy_board_count_delta: post.enemy_board_count - pre.enemy_board_count,
    }
}

pub fn build_history_event_v5(
    pre: &KernelState,
    post: &KernelState,
    player_id: i32,
    action_id: usize,
    action_type: String,
) -> KernelHistoryEvent {
    let turn_number = post.turn_number;
    let pre = RewardSnapshotV5::from_state(pre, player_id);
    let post = RewardSnapshotV5::from_state(post, player_id);
    let pre_board_delta = pre.my_board_power - pre.enemy_board_power;
    let post_board_delta = post.my_board_power - post.enemy_board_power;
    KernelHistoryEvent {
        actor_id: player_id,
        action_id,
        action_type,
        enemy_hero_hp_delta: pre.enemy_hero_hp - post.enemy_hero_hp,
        own_hero_hp_delta: pre.my_hero_hp - post.my_hero_hp,
        my_board_count_delta: post.my_board_count - pre.my_board_count,
        enemy_board_count_delta: post.enemy_board_count - pre.enemy_board_count,
        board_power_delta: post_board_delta - pre_board_delta,
        turn_number,
        source_card: None,
        target_card: None,
    }
}

pub fn compute_weighted_reward_v5(
    base_reward: f32,
    components: RewardComponentsV5,
    info_mode: InfoModeV5,
) -> f32 {
    const HP_POTENTIAL_DELTA: f32 = 0.08;
    const BOARD_POWER_DELTA: f32 = 0.015;
    const BOARD_POWER_DELTA_NORMALIZER: f32 = 100.0;
    const BOARD_UNDER_0_7_PENALTY: f32 = 0.008;
    const OWN_BOARD_WIPED_PENALTY: f32 = 0.015;
    const INFORMED_PENALTY_MULTIPLIER: f32 = 0.35;
    const DRAW_ASSIST_PENALTY_MULTIPLIER: f32 = 0.20;
    const MAX_SHAPING_ABS: f32 = 0.06;

    let mut informed_multiplier = 1.0;
    if info_mode.enemy_hand_known || info_mode.enemy_deck_known || info_mode.enemy_deck_order_known {
        informed_multiplier += INFORMED_PENALTY_MULTIPLIER;
    }
    if info_mode.draw_assist_enabled {
        informed_multiplier +=
            DRAW_ASSIST_PENALTY_MULTIPLIER * info_mode.clipped_draw_assist_strength();
    }

    let mut shaping = 0.0_f32;
    shaping += HP_POTENTIAL_DELTA * components.hp_potential_delta;
    shaping +=
        BOARD_POWER_DELTA * components.board_power_delta / BOARD_POWER_DELTA_NORMALIZER.max(1.0);
    if components.board_under_0_7 {
        shaping -= BOARD_UNDER_0_7_PENALTY * informed_multiplier;
    }
    if components.own_board_wiped {
        shaping -= OWN_BOARD_WIPED_PENALTY * informed_multiplier;
    }
    base_reward + shaping.clamp(-MAX_SHAPING_ABS, MAX_SHAPING_ABS)
}

pub fn compute_trainv2_reward(pre: &KernelState, post: &KernelState, actor_id: i32) -> f32 {
    if post.status == "p1_win" {
        return if actor_id == post.p1.user_id {
            1.0
        } else {
            -1.0
        };
    }
    if post.status == "p2_win" {
        return if actor_id == post.p2.user_id {
            1.0
        } else {
            -1.0
        };
    }
    if post.status == "draw" {
        return 0.0;
    }

    let (pre_me, pre_enemy) = pre.players_for(actor_id);
    let (post_me, post_enemy) = post.players_for(actor_id);
    let mut reward = 0.0_f32;

    let enemy_hp_delta = pre_enemy.hero.hp - post_enemy.hero.hp;
    if enemy_hp_delta > 0 {
        reward += 0.02 * enemy_hp_delta as f32;
    }

    let own_hp_delta = pre_me.hero.hp - post_me.hero.hp;
    if own_hp_delta > 0 {
        reward -= 0.01 * own_hp_delta as f32;
    }

    let enemy_killed = pre_enemy.board.len() as i32 - post_enemy.board.len() as i32;
    if enemy_killed > 0 {
        reward += 0.03 * enemy_killed as f32;
    }

    let own_killed = pre_me.board.len() as i32 - post_me.board.len() as i32;
    if own_killed > 0 {
        reward -= 0.02 * own_killed as f32;
    }

    let mana_spent = pre_me.mana - post_me.mana;
    if mana_spent > 0 {
        reward += (0.005 * mana_spent as f32).min(0.02);
    }

    reward
}

pub fn action_type_for_id(action_id: usize) -> &'static str {
    match decode_action_id(action_id) {
        Some(CandidateAction::EndTurn) => "end_turn",
        Some(CandidateAction::PlayCard { .. }) => "play_card",
        Some(CandidateAction::Attack { .. }) => "attack",
        None => "unknown",
    }
}

#[derive(Debug, Clone, Copy)]
struct RewardSnapshotV5 {
    my_hero_hp: i32,
    my_hero_max_hp: i32,
    enemy_hero_hp: i32,
    enemy_hero_max_hp: i32,
    my_board_count: i32,
    enemy_board_count: i32,
    my_board_power: f32,
    enemy_board_power: f32,
}

impl RewardSnapshotV5 {
    fn from_state(state: &KernelState, player_id: i32) -> Self {
        let (me, enemy) = state.players_for(player_id);
        Self {
            my_hero_hp: me.hero.hp,
            my_hero_max_hp: me.hero.max_hp.max(1),
            enemy_hero_hp: enemy.hero.hp,
            enemy_hero_max_hp: enemy.hero.max_hp.max(1),
            my_board_count: me.board.len() as i32,
            enemy_board_count: enemy.board.len() as i32,
            my_board_power: board_power(&me.board),
            enemy_board_power: board_power(&enemy.board),
        }
    }
}

impl KernelState {
    fn players_for(&self, player_id: i32) -> (&KernelPlayer, &KernelPlayer) {
        if self.p1.user_id == player_id {
            (&self.p1, &self.p2)
        } else {
            (&self.p2, &self.p1)
        }
    }

    fn players_for_mut(
        &mut self,
        player_id: i32,
    ) -> Result<(&mut KernelPlayer, &mut KernelPlayer), String> {
        if self.p1.user_id == player_id {
            Ok((&mut self.p1, &mut self.p2))
        } else if self.p2.user_id == player_id {
            Ok((&mut self.p2, &mut self.p1))
        } else {
            Err("unknown_player".to_string())
        }
    }

    fn player_mut(&mut self, player_id: i32) -> Result<&mut KernelPlayer, String> {
        if self.p1.user_id == player_id {
            Ok(&mut self.p1)
        } else if self.p2.user_id == player_id {
            Ok(&mut self.p2)
        } else {
            Err("unknown_player".to_string())
        }
    }
}

impl KernelCard {
    fn is_warrior(&self) -> bool {
        self.card_type == "warrior"
    }

    fn is_potion(&self) -> bool {
        self.card_type == "potion"
    }

    fn has_mechanic(&self, mechanic: &str) -> bool {
        self.mechanics.iter().any(|m| m == mechanic)
    }

    fn shape(
        &self,
        board_pos: Option<usize>,
        hand_pos: Option<usize>,
        effective_attack: Option<i32>,
    ) -> [f32; CARD_SHAPE_DIM] {
        let mechanics: Vec<&str> = self.mechanics.iter().map(String::as_str).collect();
        let input = CardShapeInput {
            card_type: match self.card_type.as_str() {
                "hero" => CardType::Hero,
                "potion" => CardType::Potion,
                _ => CardType::Warrior,
            },
            mana_cost: self.mana_cost,
            attack: self.attack,
            hp: self.hp,
            max_hp: self.max_hp,
            is_ready: self.is_ready,
            is_frozen: self.is_frozen,
            level: self.level,
            mechanics: &mechanics,
        };
        encode_card_shape(Some(&input), board_pos, hand_pos, effective_attack)
    }
}

fn compute_effective_attack(unit: &KernelCard, board: &[KernelCard], hero: &KernelCard) -> i32 {
    let mut bonus = 0;
    for aura_unit in board {
        if std::ptr::eq(unit, aura_unit) {
            continue;
        }
        bonus += aura_unit
            .mechanics
            .iter()
            .filter_map(|m| parse_aura_atk(m))
            .sum::<i32>();
    }
    bonus += hero
        .mechanics
        .iter()
        .filter_map(|m| parse_aura_atk(m))
        .sum::<i32>();
    unit.attack + bonus
}

fn parse_aura_atk(mechanic: &str) -> Option<i32> {
    let rest = mechanic.strip_prefix("aura_atk_")?;
    if rest.contains('_') {
        return None;
    }
    rest.parse().ok()
}

fn requires_target(mechanics: &[String]) -> bool {
    if mechanics.iter().any(|m| m == "cast_random_spell") {
        return false;
    }
    if mechanics.iter().any(|m| {
        matches!(
            m.as_str(),
            "delete_target" | "consume_ally" | "freeze" | "battlecry_freeze"
        )
    }) {
        return true;
    }
    mechanics
        .iter()
        .filter(|m| !m.starts_with("deathrattle_"))
        .any(|m| {
            has_prefixed_number(m, "damage_")
                || has_prefixed_number(m, "battlecry_damage_")
                || has_prefixed_number(m, "heal_")
                || has_prefixed_number(m, "heal_target_")
                || has_prefixed_number(m, "battlecry_heal_")
                || has_prefixed_number(m, "battlecry_heal_target_")
                || has_prefixed_number(m, "spell_heal_")
                || has_prefixed_number(m, "spell_heal_target_")
                || has_prefixed_number(m, "spell_damage_")
                || has_prefixed_number(m, "battlecry_buff_")
                || has_prefixed_number(m, "freeze_")
        })
}

fn has_prefixed_number(value: &str, prefix: &str) -> bool {
    value
        .strip_prefix(prefix)
        .and_then(|rest| rest.split('_').next())
        .and_then(|chunk| chunk.parse::<i32>().ok())
        .is_some()
}

fn is_random_battlecry_damage_card(card: &KernelCard) -> bool {
    card.card_id == 15
}

fn apply_regen(card: &mut KernelCard) {
    if card.hp <= 0 {
        return;
    }
    let regen = card
        .mechanics
        .iter()
        .filter_map(|m| m.strip_prefix("regen_").and_then(parse_i32_prefix))
        .next();
    if let Some(amount) = regen {
        heal_card(card, amount);
    }
}

fn heal_card(card: &mut KernelCard, amount: i32) {
    card.hp = (card.hp + amount).min(card.max_hp);
}

fn apply_damage(target: &mut KernelCard, damage: i32, attacker: Option<&mut KernelCard>) -> i32 {
    let modified_damage = apply_damage_modifiers(target, damage);
    let old_hp = target.hp;
    target.hp = (target.hp - modified_damage).max(0);
    let actual_damage = (old_hp - target.hp).max(0);

    if actual_damage > 0 {
        if let Some(attacker) = attacker {
            if let Some(reflect_amount) = target
                .mechanics
                .iter()
                .filter_map(|m| m.strip_prefix("reflect_").and_then(parse_i32_prefix))
                .next()
            {
                let reflect_damage = apply_damage_modifiers(attacker, reflect_amount);
                attacker.hp = (attacker.hp - reflect_damage).max(0);
            }
        }
    }

    actual_damage
}

fn apply_damage_modifiers(target: &mut KernelCard, damage: i32) -> i32 {
    if target.has_mechanic("permanent_shield") {
        return 0;
    }
    if consume_shield(target) {
        return 0;
    }
    let armor = target
        .mechanics
        .iter()
        .filter_map(|m| m.strip_prefix("armor_").and_then(parse_i32_prefix))
        .next()
        .unwrap_or(0);
    (damage - armor).max(0)
}

fn consume_shield(target: &mut KernelCard) -> bool {
    if let Some(idx) = target.mechanics.iter().position(|m| m == "shield") {
        target.mechanics.remove(idx);
        true
    } else {
        false
    }
}

/// Mirror of `core/engine.py::_compute_draw_weights` (Phase 1 parity port).
///
/// For each card in `player.deck` returns base weight `1.0` plus:
///   * `STUCK_BONUS * skip_count` — anti-stuck bonus for cards lingering in
///     the deck,
///   * `COST_BIAS` if the cheap/expensive bucket is underrepresented in hand.
fn compute_draw_weights(player: &KernelPlayer) -> Vec<f32> {
    let cheap_in_hand = player
        .hand
        .iter()
        .filter(|h| h.mana_cost <= CHEAP_COST_MAX)
        .count() as i32;
    let expensive_in_hand = player
        .hand
        .iter()
        .filter(|h| h.mana_cost >= EXPENSIVE_COST_MIN)
        .count() as i32;

    let mut weights = Vec::with_capacity(player.deck.len());
    for c in &player.deck {
        let base = 1.0_f32;
        let stuck = c.skip_count as f32 * STUCK_BONUS;
        let cost_bias = if c.mana_cost <= CHEAP_COST_MAX {
            (1 - cheap_in_hand).max(0) as f32 * COST_BIAS
        } else if c.mana_cost >= EXPENSIVE_COST_MIN {
            (1 - expensive_in_hand).max(0) as f32 * COST_BIAS
        } else {
            0.0
        };
        weights.push(base + stuck + cost_bias);
    }
    weights
}

/// Mirror of `core/engine.py::_weighted_choice_idx`.
///
/// `total = sum(weights)`; if `total <= 0` returns `0`. Otherwise
/// `target = rng.gen::<f64>() * total` and returns the first index whose
/// running cumulative weight exceeds `target`. Falls back to
/// `len(weights) - 1`.
///
/// Recorded-outcome RNG (task #14): when `draw_rng` is `Recorded`, the next
/// recorded pick index is popped directly — no RNG draw. This is the
/// outcome-based replay path that closes DW-7 for multi-card decks (the
/// `Deterministic` zero-RNG only matched Python for single-card decks).
fn weighted_choice_idx(weights: &[f32], draw_rng: &mut DrawRng) -> usize {
    if draw_rng.is_recorded() {
        return draw_rng.next_pick();
    }
    let DrawRng::Live(rng) = draw_rng else {
        unreachable!()
    };
    let total: f32 = weights.iter().copied().sum();
    if total <= 0.0 {
        return 0;
    }
    let target = rng.gen::<f64>() * total as f64;
    let mut cumulative = 0.0_f64;
    for (i, w) in weights.iter().copied().enumerate() {
        cumulative += w as f64;
        if cumulative > target {
            return i;
        }
    }
    weights.len().saturating_sub(1)
}

/// Mirror of `CardInstance.reset_to_base_state` (core/state.py).
///
/// Ensures the base snapshot is captured on first call (base_attack /
/// base_max_hp / base_hp / base_mana_cost / base_mechanics), then restores
/// the card to that snapshot: attack/max_hp/hp/mana_cost/mechanics reset,
/// is_ready/is_frozen/instant_kill_used false, skip_count 0.
fn reset_to_base_state(card: &mut KernelCard) {
    if !card.base_snapshot_set {
        card.base_attack = card.attack;
        card.base_max_hp = card.max_hp;
        // Python: base_hp = max_hp if max_hp else hp
        card.base_hp = if card.max_hp != 0 { card.max_hp } else { card.hp };
        card.base_mana_cost = card.mana_cost;
        card.base_mechanics = card.mechanics.clone();
        card.base_snapshot_set = true;
    }
    card.attack = card.base_attack;
    card.max_hp = card.base_max_hp;
    card.hp = if card.base_hp != 0 { card.base_hp } else { card.max_hp };
    card.mana_cost = card.base_mana_cost;
    card.mechanics = card.base_mechanics.clone();
    card.is_ready = false;
    card.is_frozen = false;
    card.skip_count = 0;
}

/// Mirror of `core/engine.py::draw_one_from_deck` (Phase 1 parity port).
///
/// 1. If deck empty: reshuffle graveyard into deck (reset_to_base_state on
///    each graveyard card, rng.shuffle). If graveyard also empty: return
///    false (fatigue).
/// 2. Increment skip_count for every card in deck.
/// 3. If hand at HAND_CAP: if overdraw_to_discard, weighted-pick a card,
///    move it to graveyard (skip_count reset + compensate remaining);
///    return true. Else return false (top card stays in deck).
/// 4. Clean draw: weighted-pick a card, move to hand (skip_count reset +
///    compensate remaining); return true.
///
/// Recorded-outcome RNG (task #14): when `draw_rng` is `Recorded`, the
/// reshuffle reorders the graveyard to match the recorded post-shuffle
/// `card_id` order (popped from `draw_rng`) instead of calling `shuffle`.
/// This is the outcome-based replay path — the graveyard's current
/// `card_id`s equal Python's pre-shuffle order (state matches), so a
/// stable first-match assignment reproduces the recorded deck order. For
/// duplicate `card_id`s, first-match preserves relative order (the
/// duplicate instances are identical after `reset_to_base_state`).
fn draw_one_from_deck(
    player: &mut KernelPlayer,
    overdraw_to_discard: bool,
    draw_rng: &mut DrawRng,
) -> bool {
    // Step 1: empty deck → reshuffle from graveyard.
    if player.deck.is_empty() {
        if !player.graveyard.is_empty() {
            for card in &mut player.graveyard {
                reset_to_base_state(card);
            }
            if draw_rng.is_recorded() {
                // Recorded-outcome reshuffle: reorder the graveyard to match
                // the recorded post-shuffle card_id order.
                if let Some(order) = draw_rng.next_order() {
                    let mut remaining = std::mem::take(&mut player.graveyard);
                    let mut deck = Vec::with_capacity(order.len());
                    for cid in &order {
                        if let Some(pos) = remaining.iter().position(|c| c.card_id == *cid) {
                            deck.push(remaining.remove(pos));
                        }
                    }
                    // Any leftover (unmatched) cards are appended preserving
                    // relative order — a desync safety net; should not happen
                    // when state matches the fixture.
                    deck.extend(remaining);
                    player.deck = deck;
                    player.graveyard.clear();
                } else {
                    // Stream exhausted: fall back to identity (graveyard
                    // order as-is) — matches Deterministic no-shuffle for
                    // single-card graveyards.
                    player.deck = std::mem::take(&mut player.graveyard);
                }
            } else {
                let DrawRng::Live(rng) = draw_rng else {
                    unreachable!()
                };
                let mut graveyard_cards = player.graveyard.clone();
                graveyard_cards.shuffle(rng);
                player.deck = graveyard_cards;
                player.graveyard.clear();
            }
        } else {
            return false;
        }
    }

    // Step 2: age every deck card (skip_count += 1).
    for c in &mut player.deck {
        c.skip_count += 1;
    }

    // Step 3: hand cap / overdraw.
    if player.hand.len() >= HAND_CAP {
        if overdraw_to_discard {
            let weights = compute_draw_weights(player);
            let choice_idx = weighted_choice_idx(&weights, draw_rng);
            let mut overdrawn = player.deck.remove(choice_idx);
            overdrawn.skip_count = 0;
            for c in &mut player.deck {
                c.skip_count = (c.skip_count - 1).max(0);
            }
            player.graveyard.push(overdrawn);
            return true;
        }
        return false;
    }

    // Step 4: clean draw.
    let weights = compute_draw_weights(player);
    let choice_idx = weighted_choice_idx(&weights, draw_rng);
    let mut drawn = player.deck.remove(choice_idx);
    drawn.skip_count = 0;
    for c in &mut player.deck {
        c.skip_count = (c.skip_count - 1).max(0);
    }
    player.hand.push(drawn);
    true
}

fn cleanup_dead_units(state: &mut KernelState) {
    loop {
        let before_p1 = state.p1.board.len();
        let before_p2 = state.p2.board.len();
        {
            let (p1, p2) = (&mut state.p1, &mut state.p2);
            cleanup_dead_units_for_player(p1, p2);
            cleanup_dead_units_for_player(p2, p1);
        }
        if state.p1.board.len() == before_p1 && state.p2.board.len() == before_p2 {
            break;
        }
    }
}

fn cleanup_dead_units_for_player(player: &mut KernelPlayer, opponent: &mut KernelPlayer) {
    let mut alive = Vec::with_capacity(player.board.len());
    for unit in player.board.drain(..) {
        if unit.hp <= 0 {
            apply_deathrattle_effects(&unit, opponent);
            player.graveyard.push(unit);
        } else {
            alive.push(unit);
        }
    }
    player.board = alive;
}

fn apply_deathrattle_effects(unit: &KernelCard, opponent: &mut KernelPlayer) {
    for mechanic in &unit.mechanics {
        let Some(amount) = mechanic
            .strip_prefix("deathrattle_aoe_damage_")
            .and_then(parse_i32_prefix)
        else {
            continue;
        };
        for target in &mut opponent.board {
            if target.hp > 0 {
                apply_damage(target, amount, None);
            }
        }
        if opponent.hero.hp > 0 {
            apply_damage(&mut opponent.hero, amount, None);
        }
        break;
    }
}

fn check_game_over(state: &mut KernelState) {
    let p1_dead = state.p1.hero.hp <= 0;
    let p2_dead = state.p2.hero.hp <= 0;
    state.status = if p1_dead && p2_dead {
        "draw".to_string()
    } else if p1_dead {
        "p2_win".to_string()
    } else if p2_dead {
        "p1_win".to_string()
    } else {
        "ongoing".to_string()
    };
}

fn board_power(board: &[KernelCard]) -> f32 {
    board
        .iter()
        .map(|card| card.attack.max(0) * card.hp.max(0))
        .sum::<i32>() as f32
}

fn missing_ratio(hp: i32, max_hp: i32) -> f32 {
    ((max_hp - hp).max(0) as f32 / max_hp.max(1) as f32).clamp(0.0, 1.0)
}

fn norm(value: f32, divisor: f32) -> f32 {
    ((value as f64 / divisor as f64).min(1.0)) as f32
}

fn norm64(value: f64, divisor: f64) -> f32 {
    (value / divisor).min(1.0) as f32
}

fn signed_norm(value: f32, divisor: f32) -> f32 {
    (value / divisor).clamp(-1.0, 1.0)
}

fn bool_f32(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn default_true() -> bool {
    true
}

fn default_mana_per_turn() -> i32 {
    1
}

fn parse_i32_prefix(value: &str) -> Option<i32> {
    value.split('_').next()?.parse::<i32>().ok()
}

#[cfg(test)]
mod draw_tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    fn card(id: i32, mana_cost: i32, attack: i32, hp: i32) -> KernelCard {
        KernelCard {
            card_id: id,
            card_type: "warrior".to_string(),
            mana_cost,
            attack,
            hp,
            max_hp: hp,
            mechanics: Vec::new(),
            is_ready: false,
            is_frozen: false,
            level: 1,
            ..Default::default()
        }
    }

    fn player_with(deck: Vec<KernelCard>, hand: Vec<KernelCard>) -> KernelPlayer {
        KernelPlayer {
            user_id: 1,
            mana: 10,
            max_mana: 10,
            hero: card(0, 0, 0, 30),
            hand,
            board: Vec::new(),
            deck,
            graveyard: Vec::new(),
            trophies: 0,
            mana_draw_count_this_turn: 0,
        }
    }

    #[test]
    fn compute_draw_weights_matches_python_semantics() {
        // deck: cheap (cost 2), mid (cost 3), expensive (cost 4).
        // hand: 0 cheap, 0 expensive → cheap card gets COST_BIAS, expensive gets COST_BIAS.
        let p = player_with(
            vec![card(11, 2, 1, 1), card(12, 3, 1, 1), card(13, 4, 1, 1)],
            vec![card(20, 3, 1, 1)],
        );
        let w = compute_draw_weights(&p);
        // base 1.0, skip_count 0 → no stuck. cheap_in_hand=0, expensive_in_hand=0.
        // cost 2 → 1.0 + max(0,1-0)*0.3 = 1.3
        // cost 3 → 1.0 + 0 = 1.0
        // cost 4 → 1.0 + max(0,1-0)*0.3 = 1.3
        assert_eq!(w, vec![1.3, 1.0, 1.3]);

        // With a cheap card in hand, cheap bucket satisfied → cheap card loses bias.
        let p2 = player_with(
            vec![card(11, 2, 1, 1), card(12, 3, 1, 1), card(13, 4, 1, 1)],
            vec![card(20, 2, 1, 1), card(21, 4, 1, 1)],
        );
        let w2 = compute_draw_weights(&p2);
        // cheap_in_hand=1, expensive_in_hand=1 → both biases 0.
        assert_eq!(w2, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn weighted_choice_idx_zero_total_returns_zero() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        assert_eq!(weighted_choice_idx(&[0.0, 0.0, 0.0], &mut draw_rng), 0);
    }

    #[test]
    fn weighted_choice_idx_picks_index_zero_with_zero_rng() {
        // WorkerRng::Deterministic returns 0 for next_u64 → gen::<f64>() == 0.0.
        let mut rng = crate::worker::WorkerRng::Deterministic;
        // target = 0.0 * total = 0.0; cumulative[0] = 1.0 > 0 → idx 0.
        let mut draw_rng = DrawRng::live(&mut rng);
        assert_eq!(weighted_choice_idx(&[1.0, 1.0, 1.0, 1.0], &mut draw_rng), 0);
        assert_eq!(weighted_choice_idx(&[0.5, 2.0, 1.5], &mut draw_rng), 0);
    }

    #[test]
    fn draw_one_from_deck_clean_draw_picks_and_resets_skip_count() {
        let mut p = player_with(
            vec![card(11, 2, 1, 1), card(12, 3, 1, 1), card(13, 4, 1, 1)],
            vec![card(20, 3, 1, 1)],
        );
        // Age the deck so skip_count > 0 before the draw.
        for c in &mut p.deck {
            c.skip_count = 2;
        }
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let drew = draw_one_from_deck(&mut p, false, &mut draw_rng);
        assert!(drew);
        // Deterministic rng picks idx 0 → card 11 removed.
        assert_eq!(p.deck.iter().map(|c| c.card_id).collect::<Vec<_>>(), vec![12, 13]);
        // Drawn card's skip_count reset to 0.
        assert_eq!(p.hand.last().unwrap().skip_count, 0);
        // Remaining deck cards compensated: skip_count 2 → 1 (after +1 then -1 net 0? see impl).
        // Impl: +1 (age) → 3, then compensate -1 → 2. So skip_count stays 2.
        assert_eq!(p.deck[0].skip_count, 2);
        assert_eq!(p.hand.len(), 2);
    }

    #[test]
    fn draw_one_from_deck_hand_full_no_overdraw_keeps_card() {
        // hand at HAND_CAP (4), overdraw_to_discard=false → card stays in deck.
        let hand = vec![
            card(20, 1, 1, 1),
            card(21, 2, 1, 1),
            card(22, 3, 1, 1),
            card(23, 4, 1, 1),
        ];
        let mut p = player_with(vec![card(11, 2, 1, 1), card(12, 3, 1, 1)], hand);
        let deck_before = p.deck.iter().map(|c| c.card_id).collect::<Vec<_>>();
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let drew = draw_one_from_deck(&mut p, false, &mut draw_rng);
        assert!(!drew);
        assert_eq!(p.deck.iter().map(|c| c.card_id).collect::<Vec<_>>(), deck_before);
        assert_eq!(p.hand.len(), HAND_CAP);
    }

    #[test]
    fn draw_one_from_deck_overdraw_to_discard_moves_to_graveyard() {
        let hand = vec![
            card(20, 1, 1, 1),
            card(21, 2, 1, 1),
            card(22, 3, 1, 1),
            card(23, 4, 1, 1),
        ];
        let mut p = player_with(vec![card(11, 2, 1, 1), card(12, 3, 1, 1)], hand);
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let drew = draw_one_from_deck(&mut p, true, &mut draw_rng);
        assert!(drew);
        // Deterministic picks idx 0 → card 11 to graveyard.
        assert_eq!(p.deck.iter().map(|c| c.card_id).collect::<Vec<_>>(), vec![12]);
        assert_eq!(p.graveyard.len(), 1);
        assert_eq!(p.graveyard[0].card_id, 11);
        assert_eq!(p.hand.len(), HAND_CAP);
    }

    #[test]
    fn draw_one_from_deck_reshuffle_from_graveyard_when_deck_empty() {
        let mut p = player_with(Vec::new(), vec![card(20, 3, 1, 1)]);
        // Put a modified card in graveyard; reshuffle must reset_to_base_state.
        let mut gy_card = card(11, 2, 5, 1);
        gy_card.attack = 9; // modified at runtime
        gy_card.base_attack = 2; // snapshot already taken (base_snapshot_set true)
        gy_card.base_max_hp = 1;
        gy_card.base_hp = 1;
        gy_card.base_mana_cost = 2;
        gy_card.base_mechanics = vec!["charge".to_string()];
        gy_card.base_snapshot_set = true;
        p.graveyard = vec![gy_card];
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let drew = draw_one_from_deck(&mut p, false, &mut draw_rng);
        assert!(drew);
        // After reshuffle the single graveyard card is back in deck then drawn into hand.
        assert_eq!(p.hand.len(), 2);
        assert_eq!(p.hand[1].card_id, 11);
        // reset_to_base_state restored attack from base_attack (2), not 9.
        assert_eq!(p.hand[1].attack, 2);
        assert_eq!(p.graveyard.len(), 0);
    }

    #[test]
    fn draw_one_from_deck_fatigue_when_deck_and_graveyard_empty() {
        let mut p = player_with(Vec::new(), vec![card(20, 3, 1, 1)]);
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let drew = draw_one_from_deck(&mut p, false, &mut draw_rng);
        assert!(!drew);
        assert_eq!(p.hand.len(), 1);
    }

    // ---- Phase 2: board cap (AC-FFI-1) ----

    #[test]
    fn game_board_cap_is_five_and_distinct_from_codec_num_board() {
        assert_eq!(GAME_BOARD_CAP, 5);
        // The frozen codec capacity stays 7 (slot layout), independent of the
        // 5-card game rule.
        assert_eq!(crate::action_codec::NUM_BOARD, 7);
    }

    // ---- Phase 2: mana_draw (MD-1/2/3/5) ----

    #[test]
    fn mana_draw_legal_for_requires_hand_below_cap_and_mana_for_next_cost() {
        let mut p = player_with(vec![card(99, 1, 1, 1)], vec![card(20, 3, 1, 1)]);
        // mana=10, hand=1, count=0 → cost 2 → legal.
        assert!(mana_draw_legal_for(&p));
        // hand full (4) → illegal.
        p.hand = vec![
            card(20, 1, 1, 1),
            card(21, 1, 1, 1),
            card(22, 1, 1, 1),
            card(23, 1, 1, 1),
        ];
        assert!(!mana_draw_legal_for(&p));
        // mana below next cost → illegal.
        p.hand = vec![card(20, 3, 1, 1)];
        p.mana = 1;
        p.mana_draw_count_this_turn = 0;
        assert!(!mana_draw_legal_for(&p)); // cost 2 > mana 1
        // count raises the cost → illegal once mana can't cover it.
        p.mana = 3;
        p.mana_draw_count_this_turn = 1; // cost = 2*(1+1) = 4 > 3
        assert!(!mana_draw_legal_for(&p));
        p.mana = 4;
        assert!(mana_draw_legal_for(&p)); // cost 4 == mana 4
    }

    #[test]
    fn apply_mana_draw_deducts_cost_draws_and_increments_count() {
        let mut p = player_with(vec![card(99, 1, 1, 1)], vec![card(20, 3, 1, 1)]);
        p.mana = 5;
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let res = apply_mana_draw(&mut p, false, &mut draw_rng);
        assert!(res.is_ok());
        assert_eq!(p.mana, 3); // 5 - 2
        assert_eq!(p.hand.len(), 2);
        assert_eq!(p.hand[1].card_id, 99);
        assert_eq!(p.mana_draw_count_this_turn, 1);
    }

    #[test]
    fn apply_mana_draw_refunds_mana_on_fatigue_and_leaves_count_unchanged() {
        // MD-5: cost deducted BEFORE draw; on draw-failure (fatigue) mana is
        // REFUNDED and the counter does NOT increment (matches Python
        // `_handle_mana_draw` `player.mana += cost`).
        let mut p = player_with(Vec::new(), vec![card(20, 3, 1, 1)]);
        p.mana = 5;
        let mana_before = p.mana;
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let res = apply_mana_draw(&mut p, false, &mut draw_rng);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "no_cards_to_draw");
        assert_eq!(p.mana, mana_before); // refunded
        assert_eq!(p.mana_draw_count_this_turn, 0); // unchanged
        assert_eq!(p.hand.len(), 1); // no draw
    }

    #[test]
    fn apply_mana_draw_cost_grows_linearly_with_count() {
        // N-th draw costs MANA_DRAW_BASE * (count + 1) = 2, 4, 6, ...
        let mut p = player_with(
            vec![card(99, 1, 1, 1), card(98, 1, 1, 1), card(97, 1, 1, 1)],
            vec![card(20, 3, 1, 1)],
        );
        p.mana = 12;
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        // 1st draw: cost 2
        assert!(apply_mana_draw(&mut p, false, &mut draw_rng).is_ok());
        assert_eq!(p.mana, 10);
        assert_eq!(p.mana_draw_count_this_turn, 1);
        // 2nd draw: cost 4
        assert!(apply_mana_draw(&mut p, false, &mut draw_rng).is_ok());
        assert_eq!(p.mana, 6);
        assert_eq!(p.mana_draw_count_this_turn, 2);
        // 3rd draw: cost 6
        assert!(apply_mana_draw(&mut p, false, &mut draw_rng).is_ok());
        assert_eq!(p.mana, 0);
        assert_eq!(p.mana_draw_count_this_turn, 3);
    }
}
