use rand::seq::SliceRandom;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};

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
        /// Per-step recorded `random.randint` outcomes (Phase 4: armor_X_Y
        /// range roll). Popped in call order by `roll_range`. Rust must call
        /// `roll_range` in exactly the same places Python calls
        /// `random.randint` so the streams stay aligned.
        randint_rolls: VecDeque<i32>,
        /// Per-step recorded `random.choice` outcomes (Phase 6: card15
        /// `battlecry_damage_X_random` random-target pick). Each entry is the
        /// 0-based index into the Python `targets` list
        /// (`list(opponent.board) + [opponent.hero]`) that `random.choice`
        /// returned. Popped in call order by `roll_choice`. Rust must call
        /// `roll_choice` in exactly the same places Python calls
        /// `random.choice` so the streams stay aligned. Default empty for
        /// pre-Phase-6 fixtures.
        choice_rolls: VecDeque<i32>,
        /// Per-step recorded `random.sample` outcomes (Phase 9: card26
        /// `cast_random_spell` Blackwhip freeze-target pick). Each entry is the
        /// list of 0-based indices into the Python `population` list
        /// (`unfrozen_enemies = [u for u in opponent.board if not
        /// u.is_frozen]`) that `random.sample(population, k)` returned, in
        /// selection order. Popped in call order by `roll_sample`. Rust must
        /// call `roll_sample` in exactly the same places Python calls
        /// `random.sample` so the streams stay aligned. Default empty for
        /// pre-Phase-9 fixtures (no `random.sample` mechanic exercised).
        sample_rolls: VecDeque<Vec<i32>>,
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
    /// `sample_rolls` defaults empty (pre-Phase-9 fixtures never exercise
    /// `random.sample`). Kept as a 4-arg constructor so existing unit tests
    /// that don't touch `cast_random_spell` stay unchanged.
    pub fn recorded(
        picks: Vec<usize>,
        orders: Vec<Vec<i32>>,
        randint_rolls: Vec<i32>,
        choice_rolls: Vec<i32>,
    ) -> Self {
        DrawRng::Recorded {
            picks: picks.into(),
            orders: orders.into(),
            randint_rolls: randint_rolls.into(),
            choice_rolls: choice_rolls.into(),
            sample_rolls: VecDeque::new(),
        }
    }

    /// Build a `Recorded` DrawRng including the `random.sample` outcome
    /// stream (Phase 9: card26 `cast_random_spell` Blackwhip). Use this for
    /// fixtures that exercise `random.sample`; otherwise prefer `recorded`.
    pub fn recorded_with_sample(
        picks: Vec<usize>,
        orders: Vec<Vec<i32>>,
        randint_rolls: Vec<i32>,
        choice_rolls: Vec<i32>,
        sample_rolls: Vec<Vec<i32>>,
    ) -> Self {
        DrawRng::Recorded {
            picks: picks.into(),
            orders: orders.into(),
            randint_rolls: randint_rolls.into(),
            choice_rolls: choice_rolls.into(),
            sample_rolls: sample_rolls.into(),
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

/// Mirror of Python `random.randint(min, max)` inclusive range roll, used by
/// `armor_X_Y` damage reduction (core/effects.py `apply_damage_modifiers`)
/// and — later, Phase 6 — card15 `cast_random_spell` spell-choice. Rust must
/// call this in the SAME code paths Python calls `random.randint` so the
/// recorded-outcome stream (`DrawRng::Recorded.randint_rolls`) replays in
/// call order.
///
/// `Recorded`: pop the next recorded int; if exhausted fall back to `min`
/// with a debug warning (keeps replay robust, mirrors the pick/order
/// exhaustion contract). `Live`: `rng.gen_range(min..=max)`.
fn roll_range(min: i32, max: i32, draw_rng: &mut DrawRng) -> i32 {
    // Python `random.randint(a, a)` (min>=max degenerate range, e.g. armor_X_X)
    // STILL consumes the RNG stream — `golden_trace._recording_randint`
    // appends unconditionally. To keep the recorded-outcome stream aligned
    // across degenerate rolls, pop the recorded value BEFORE the min>=max
    // early-return guard (Phase-4 latent: degenerate armor_X_X stream
    // misalignment). Live mode mirrors Python's unconditional `randint` call
    // by going through the gen_range path below only when min<max (a
    // degenerate live range would panic in gen_range, so keep the guard for
    // Live, but Recorded pops first).
    match draw_rng {
        DrawRng::Recorded { randint_rolls, .. } => {
            let rolled = randint_rolls.pop_front().unwrap_or_else(|| {
                eprintln!(
                    "[DrawRng] recorded randint stream exhausted; falling back to min={}",
                    min
                );
                min
            });
            if min >= max {
                // Degenerate range: Python still consumed the stream; Rust has
                // now popped it. Return min (Python's randint(a,a)==a).
                min
            } else {
                rolled.clamp(min, max)
            }
        }
        DrawRng::Live(rng) => {
            if min >= max {
                min
            } else {
                rng.gen_range(min..=max)
            }
        }
    }
}

/// Mirror of Python `random.choice(seq)` — returns a 0-based index into a
/// list of length `n`. Used by card15 `battlecry_damage_X_random`
/// (core/effects.py `_apply_random_battlecry_damage`: `random.choice(targets)`
/// where `targets = list(opponent.board) + [opponent.hero]`). Rust must call
/// this in the SAME code paths Python calls `random.choice` so the
/// recorded-outcome stream (`DrawRng::Recorded.choice_rolls`) replays in
/// call order.
///
/// `Recorded`: pop the next recorded index; if exhausted fall back to `0`
/// with a debug warning (keeps replay robust, mirrors the pick/order
/// exhaustion contract). The popped value is clamped to `[0, n)` so a stale
/// fixture entry can't index out of bounds. `Live`: `rng.gen_range(0..n)`.
fn roll_choice(n: usize, draw_rng: &mut DrawRng) -> usize {
    if n == 0 {
        return 0;
    }
    // For the Recorded path, Python's `random.choice([x])` still consumes RNG
    // via `_randbelow(1) -> getrandbits(1)`, and golden_trace `_recording_choice`
    // ALWAYS appends a choice_roll (idx 0). So we must pop `choice_rolls` even
    // when `n == 1` to keep the stream in sync with Python. The popped value is
    // discarded (clamped to 0). For the Live path, n==1 is a no-op short-circuit
    // (no RNG consumed — `gen_range(0..1)` would be a no-op anyway).
    match draw_rng {
        DrawRng::Recorded { choice_rolls, .. } => {
            let raw = choice_rolls.pop_front().unwrap_or_else(|| {
                eprintln!(
                    "[DrawRng] recorded choice stream exhausted; falling back to 0"
                );
                0
            });
            // Clamp guards against stale/out-of-range fixture entries.
            let idx = raw.clamp(0, (n - 1) as i32) as usize;
            idx
        }
        DrawRng::Live(_rng) => {
            if n == 1 {
                return 0;
            }
            _rng.gen_range(0..n)
        }
    }
}

/// Mirror of Python `random.sample(population, k)` — returns the list of
/// 0-based indices into a population of length `n` that `random.sample`
/// selected, in selection order. Used by card26 `cast_random_spell` spell 3
/// (Blackwhip): `targets_to_freeze = random.sample(unfrozen_enemies,
/// min(freeze_count, len(unfrozen_enemies)))`. Rust must call this in the
/// SAME code paths Python calls `random.sample` so the recorded-outcome
/// stream (`DrawRng::Recorded.sample_rolls`) replays in call order.
///
/// `Recorded`: pop the next recorded index-list; if exhausted fall back to a
/// deterministic prefix `[0, 1, ..., k-1]` (keeps replay robust, mirrors the
/// pick/order/choice exhaustion contract). The popped indices are clamped to
/// `[0, n)` so a stale fixture entry can't index out of bounds. `Live`: draw
/// `k` distinct indices from `0..n` using the same `StdRng` the worker uses
/// (matches Python's *distribution*, not its MT19937 stream —
/// recorded-outcome fixtures pin byte-parity, live training just needs a
/// valid sample).
fn roll_sample(n: usize, k: usize, draw_rng: &mut DrawRng) -> Vec<usize> {
    let k = k.min(n);
    if k == 0 || n == 0 {
        return Vec::new();
    }
    match draw_rng {
        DrawRng::Recorded { sample_rolls, .. } => {
            let indices = sample_rolls.pop_front().unwrap_or_else(|| {
                eprintln!(
                    "[DrawRng] recorded sample stream exhausted; falling back to prefix"
                );
                (0..k as i32).collect::<Vec<_>>()
            });
            indices
                .into_iter()
                .map(|i| (i.clamp(0, (n - 1) as i32)) as usize)
                .take(k)
                .collect()
        }
        DrawRng::Live(rng) => {
            // Fisher-Yates partial shuffle on 0..n, take the first k.
            let mut perm: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = rng.gen_range(i..n);
                perm.swap(i, j);
            }
            perm[..k].to_vec()
        }
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
    /// Sudden-death modifier (WD-1): when true, the active player's hero takes
    /// escalating damage at the start of each of their turns. Mirrors
    /// `infrastructure/match_modes.py::ClassicParams.sudden_death_enabled`.
    /// Default false — `ClassicRLEnv` does not pass `classic_params`, so the
    /// classic training env never enables sudden-death.
    #[serde(default)]
    pub sudden_death_enabled: bool,
    /// Sudden-death base damage applied on a player's first sudden-death tick.
    /// Mirrors `ClassicParams.sudden_death_damage_start` (default 1).
    #[serde(default = "default_sudden_death_damage_start")]
    pub sudden_death_damage_start: i32,
    /// Sudden-death per-tick damage escalation step. Mirrors
    /// `ClassicParams.sudden_death_damage_step` (default 1).
    #[serde(default = "default_sudden_death_damage_step")]
    pub sudden_death_damage_step: i32,
    /// Truncation turn limit (WD-2): the episode is truncated when
    /// `turn_number > max_turns`. Mirrors
    /// `ai/train_v2/classic_rl_env.py::ClassicRLEnv._max_turns` (default 80).
    #[serde(default = "default_max_turns")]
    pub max_turns: i32,
}

fn default_sudden_death_damage_start() -> i32 {
    1
}
fn default_sudden_death_damage_step() -> i32 {
    1
}
fn default_max_turns() -> i32 {
    80
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
    /// Recorded-outcome RNG (Phase 4): every `random.randint` result during
    /// this step (armor_X_Y range roll). Rust replay pops these in call order
    /// via `roll_range`. Default empty for pre-Phase-4 fixtures.
    #[serde(default)]
    pub randint_rolls: Vec<i32>,
    /// Recorded-outcome RNG (Phase 6): every `random.choice` result during
    /// this step, as the 0-based index into the Python `targets` list (card15
    /// `battlecry_damage_X_random` random-target pick). Rust replay pops
    /// these in call order via `roll_choice`. Default empty for pre-Phase-6
    /// fixtures (no `random.choice` mechanic exercised → no replay needed).
    #[serde(default)]
    pub choice_rolls: Vec<i32>,
    /// Recorded-outcome RNG (Phase 9): every `random.sample` result during
    /// this step, as the list of 0-based indices into the Python `population`
    /// list (card26 `cast_random_spell` Blackwhip freeze-target pick:
    /// `unfrozen_enemies = [u for u in opponent.board if not u.is_frozen]`).
    /// Rust replay pops these in call order via `roll_sample`. Default empty
    /// for pre-Phase-9 fixtures (no `random.sample` mechanic exercised).
    #[serde(default)]
    pub sample_rolls: Vec<Vec<i32>>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct KernelState {
    pub current_turn_owner_id: i32,
    pub turn_number: i32,
    pub status: String,
    pub p1: KernelPlayer,
    pub p2: KernelPlayer,
    /// Sudden-death per-player escalating damage counter — mirrors
    /// `core/state.py::GameState.sudden_death_turns_by_player`. Maps
    /// `user_id` → number of times sudden-death has been applied to that
    /// player. Empty when sudden-death is disabled (the default). The damage
    /// applied at a player's turn-Nth sudden-death tick is
    /// `sudden_death_damage_start + (N-1)*sudden_death_damage_step`
    /// (`core/engine.py::_apply_start_turn_mode_effects`).
    #[serde(default)]
    pub sudden_death_turns_by_player: BTreeMap<i32, i32>,
    /// Mirrors `core/state.py::GameState.sudden_death_last_applied_turn_by_player`.
    /// Maps `user_id` → the `turn_number` at which sudden-death was last
    /// applied to that player (guards against double-application within the
    /// same turn). Empty when sudden-death is disabled.
    #[serde(default)]
    pub sudden_death_last_applied_turn_by_player: BTreeMap<i32, i32>,
    /// Mirrors `core/state.py:183::GameState.pending_mana_drain_by_player`.
    /// Maps opponent `user_id` → mana scheduled to be drained at the START of
    /// that opponent's next turn (after their mana is restored to max_mana).
    /// Two-stage mana_drain (core/effects.py:540-580 schedules the pending
    /// overflow during play; core/engine.py:700-703 pops & applies it inside
    /// `_handle_end_turn` after `opponent.mana = opponent.max_mana`). BTreeMap
    /// for deterministic iteration/serialization. Empty by default; old
    /// fixtures have no such field so `#[serde(default)]` keeps them parsing.
    #[serde(default)]
    pub pending_mana_drain_by_player: BTreeMap<i32, i32>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
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
    /// One-shot instant_kill flag (core/state.py `CardInstance.instant_kill_used`).
    /// NOT serialized into the golden-trace state payload — Python's
    /// `_card_payload` omits it, so `skip_serializing` keeps the JSON matcher
    /// byte-stable for old fixtures (rebirth/cap use Сайтама but the field is
    /// invisible on both sides). The flag still drives Rust attack logic and
    /// is reset in `reset_to_base_state`.
    #[serde(default, skip_serializing)]
    pub instant_kill_used: bool,
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
    /// Sudden-death modifier (WD-1). Default false.
    pub sudden_death_enabled: bool,
    pub sudden_death_damage_start: i32,
    pub sudden_death_damage_step: i32,
    /// Truncation turn limit (WD-2). Default 80.
    pub max_turns: i32,
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
            sudden_death_enabled: false,
            sudden_death_damage_start: 1,
            sudden_death_damage_step: 1,
            max_turns: 80,
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
            sudden_death_enabled: config.sudden_death_enabled,
            sudden_death_damage_start: config.sudden_death_damage_start,
            sudden_death_damage_step: config.sudden_death_damage_step,
            max_turns: config.max_turns,
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
    /// Truncation flag (WD-2): true when `state.turn_number > max_turns`,
    /// mirroring `ai/train_v2/classic_rl_env.py::ClassicRLEnv.step`
    /// (`truncated = st.turn_number > self._max_turns`). Independent of
    /// `terminated` (which is `status != "ongoing"`), matching Python.
    pub truncated: bool,
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
            cleanup_dead_units(&mut next, draw_rng);
            check_game_over(&mut next);
            let base_reward = compute_trainv2_reward(state, &next, player_id);
            let reward_components_v5 = compute_reward_components_v5(state, &next, player_id);
            let reward = if self.config.v5_weighted_reward {
                compute_weighted_reward_v5(base_reward, reward_components_v5, self.config.info_mode)
            } else {
                base_reward
            };
            let terminated = next.status != "ongoing";
            let truncated = next.turn_number > self.config.max_turns;
            return Ok(KernelStepOutput {
                reward_components_v5,
                reward,
                terminated,
                truncated,
                state: next,
            });
        }

        let mask = build_action_mask(state, player_id, self.config.placement_mode);
        if action_id >= mask.len() || mask[action_id] != 1.0 {
            return Err("illegal_action".to_string());
        }

        self.apply_decoded(state, player_id, action_id, draw_rng)
    }

    /// Force-apply a (possibly mask-illegal) action via the engine apply path,
    /// BYPASSING the 601 action_mask legality check. Test-only — used by the
    /// consume_ally_full state-transition parity test: the frozen codec mask
    /// (classic_actions_v1._mask_play_actions) masks the consume_ally play OUT
    /// at a full board (no exemption), but core/engine.py:1228 + the Rust
    /// apply_play_card board-full guard DO exempt consume_ally, so the play
    /// succeeds when forced. `mana_draw_flag` MUST be false (the unchecked
    /// path is for forced 601-action plays, never the parallel mana_draw head).
    pub fn apply_action_unchecked(
        &self,
        state: &KernelState,
        player_id: i32,
        action_id: usize,
        draw_rng: &mut DrawRng,
    ) -> Result<KernelStepOutput, String> {
        if state.status != "ongoing" {
            return Err("game_over".to_string());
        }
        if state.current_turn_owner_id != player_id {
            return Err("not_your_turn".to_string());
        }
        self.apply_decoded(state, player_id, action_id, draw_rng)
    }

    /// Post-decode apply body shared by `apply_action` (mask-checked) and
    /// `apply_action_unchecked` (mask-bypassed). Decodes `action_id`, applies
    /// it via the engine apply path (apply_end_turn / apply_play_card /
    /// apply_attack), runs cleanup_dead_units + check_game_over, and computes
    /// the reward + v5 components.
    fn apply_decoded(
        &self,
        state: &KernelState,
        player_id: i32,
        action_id: usize,
        draw_rng: &mut DrawRng,
    ) -> Result<KernelStepOutput, String> {
        let mut next = state.clone();
        let overdraw_to_discard = self.config.overdraw_to_discard;
        match decode_action_id(action_id) {
            Some(CandidateAction::EndTurn) => apply_end_turn(
                &mut next,
                player_id,
                self.config.mana_per_turn,
                overdraw_to_discard,
                self.config.sudden_death_enabled,
                self.config.sudden_death_damage_start,
                self.config.sudden_death_damage_step,
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
            }) => apply_attack(&mut next, player_id, attacker_index, target_code, draw_rng)?,
            None => return Err("unknown_action".to_string()),
        }

        cleanup_dead_units(&mut next, draw_rng);
        check_game_over(&mut next);

        let base_reward = compute_trainv2_reward(state, &next, player_id);
        let reward_components_v5 = compute_reward_components_v5(state, &next, player_id);
        let reward = if self.config.v5_weighted_reward {
            compute_weighted_reward_v5(base_reward, reward_components_v5, self.config.info_mode)
        } else {
            base_reward
        };
        let terminated = next.status != "ongoing";
        let truncated = next.turn_number > self.config.max_turns;

        Ok(KernelStepOutput {
            reward_components_v5,
            reward,
            terminated,
            truncated,
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
        // MASK guard: mirrors the frozen codec
        // `classic_actions_v1._mask_play_actions` (line ~228):
        //   `if is_warrior and len(me.board) >= _NUM_BOARD: continue`
        // — NO consume_ally exemption. The frozen mask masks the consume_ally
        // play OUT at a full board (the model never sees it), even though the
        // APPLY path (apply_play_card below) and core/engine.py:1228 DO exempt
        // consume_ally (the consumed ally is removed before the new card is
        // placed, so net board size stays 5). Rust's action_mask must match the
        // frozen codec byte-for-byte; the apply path mirrors engine.py:1228.
        // Phase-2: CLN-3 mask parity (reverted the prior incorrect exemption).
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
    // TAMHP (card 52): target_ally_max_hp_plus[_universal]_N. The universal
    // variant allows the own-hero target (code 16); the bare variant forbids
    // hero. Both share friendly-minion target codes 9..15. Mirrors
    // core/engine.py `get_valid_targets` branches for is_max_hp_plus_universal
    // / is_max_hp_plus. Strictly additive: only cards whose mechanics contain
    // `target_ally_max_hp_plus` (card 52) reach this branch; all other cards
    // have is_max_hp_plus_* = false and skip it, leaving existing masks
    // byte-identical (frozen-classic guard).
    let is_max_hp_plus_universal = mechanics
        .iter()
        .any(|m| m.starts_with("target_ally_max_hp_plus_universal"));
    let is_max_hp_plus = mechanics
        .iter()
        .any(|m| m.starts_with("target_ally_max_hp_plus_") && !m.starts_with("target_ally_max_hp_plus_universal"));

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

    // TAMHP universal (card 52): friendly board minions (9..15) + own hero
    // (16). Mirrors core/engine.py `get_valid_targets` is_max_hp_plus_universal
    // branch. Additive: only card 52 reaches here.
    if is_max_hp_plus_universal {
        for target_idx in 0..me.board.len() {
            mask[base + 9 + target_idx] = 1.0;
        }
        mask[base + 16] = 1.0;
        return;
    }

    // TAMHP non-universal: friendly board minions only (no hero). Mirrors
    // core/engine.py is_max_hp_plus branch. No card currently uses the bare
    // variant, but the branch is symmetric with the universal one.
    if is_max_hp_plus {
        for target_idx in 0..me.board.len() {
            mask[base + 9 + target_idx] = 1.0;
        }
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
    sudden_death_enabled: bool,
    sudden_death_damage_start: i32,
    sudden_death_damage_step: i32,
    draw_rng: &mut DrawRng,
) -> Result<(), String> {
    let next_player_id = {
        let (_player, opponent) = state.players_for(player_id);
        opponent.user_id
    };
    state.current_turn_owner_id = next_player_id;
    state.turn_number += 1;

    // Sudden-death (WD-1): mirrors core/engine.py::_apply_start_turn_mode_effects
    // invoked at the start of the active player's turn right after the
    // turn_number increment + owner switch. The active player's hero takes
    // escalating damage: damage = damage_start + (turn_count-1)*damage_step,
    // applied via DIRECT hp subtraction (not apply_damage — bypasses
    // armor/reflect, exactly like Python `player.hero.hp -= damage`). The
    // per-player counter escalates each of the player's own turns. If the
    // hero dies, check_game_over flips status and we return early WITHOUT
    // restoring mana / readying the board / drawing (matching Python's
    // `if self.state.status != GameStatus.ONGOING: return`).
    if sudden_death_enabled && state.status == "ongoing" {
        let last_applied = state
            .sudden_death_last_applied_turn_by_player
            .get(&next_player_id)
            .copied();
        if last_applied != Some(state.turn_number) {
            let turn_count =
                state.sudden_death_turns_by_player.get(&next_player_id).copied().unwrap_or(0) + 1;
            state
                .sudden_death_turns_by_player
                .insert(next_player_id, turn_count);
            state
                .sudden_death_last_applied_turn_by_player
                .insert(next_player_id, state.turn_number);
            let damage = sudden_death_damage_start + (turn_count - 1) * sudden_death_damage_step;
            if damage > 0 {
                let opponent = state.player_mut(next_player_id)?;
                opponent.hero.hp -= damage;
                check_game_over(state);
            }
        }
        if state.status != "ongoing" {
            return Ok(());
        }
    }

    // Mana-drain pending pop (mirrors core/engine.py:700-703, inside
    // `_handle_end_turn`): after `opponent.mana = opponent.max_mana` the
    // scheduled drain (recorded by mana_drain_X during the owner's play,
    // core/effects.py:540-580) is applied. Popped BEFORE the opponent mutable
    // borrow so the disjoint `state.pending_mana_drain_by_player` field borrow
    // does not conflict with `state.player_mut`.
    let pending_mana_drain = state
        .pending_mana_drain_by_player
        .remove(&next_player_id)
        .unwrap_or(0);

    let opponent = state.player_mut(next_player_id)?;
    opponent.max_mana = (opponent.max_mana + mana_per_turn).min(10);
    opponent.mana = opponent.max_mana;
    // mana_draw_count_this_turn resets at the start of each player's turn
    // (mirrors core/engine.py _handle_end_turn).
    opponent.mana_draw_count_this_turn = 0;
    // Apply the popped pending mana drain now (after restore), mirroring
    // core/engine.py:701-703: `opponent.mana = max(0, opponent.mana -
    // pending_mana_drain)`.
    if pending_mana_drain > 0 {
        opponent.mana = (opponent.mana - pending_mana_drain).max(0);
    }

    for unit in &mut opponent.board {
        if unit.is_frozen {
            unit.is_ready = false;
            unit.is_frozen = false;
        } else {
            unit.is_ready = true;
        }
        // shield_refresh (core/engine.py line 728): at the start of the
        // owner's turn, a unit with `shield_refresh` and NO current `shield`
        // regains its one-time shield. Mechanic-driven (not card_id) so any
        // card with `shield_refresh` (e.g. card24 "Годжо Сатору") behaves
        // correctly — future-proof + matches Python.
        if unit.has_mechanic("shield_refresh") && !unit.has_mechanic("shield") {
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
    // Inline field access (instead of `state.players_for_mut`) so the
    // disjoint `state.pending_mana_drain_by_player` field can be mutably
    // borrowed simultaneously and threaded into `apply_play_effects` for the
    // mana_drain branch (mirrors core/effects.py:540-580). NLL tracks the p1/p2
    // and pending_mana_drain_by_player field borrows as disjoint.
    let (player, opponent) = if state.p1.user_id == player_id {
        (&mut state.p1, &mut state.p2)
    } else if state.p2.user_id == player_id {
        (&mut state.p2, &mut state.p1)
    } else {
        return Err("unknown_player".to_string());
    };
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
        // consume_ally (Phase 3: CLN-3). Mirrors core/engine.py
        // _handle_play_card: consume the targeted friendly ally BEFORE
        // placing the new card on board. The consumed unit's current
        // attack/hp/max_hp are added to the played card; the consumed unit
        // is removed from board and sent to graveyard. Its deathrattle does
        // NOT fire (consume = "devour", not "kill" — by design). Target_code
        // 9..=15 maps to friendly board index 0..6 (same encoding as
        // mask_targets_for_card / apply_damage_to_play_target). The card's
        // base_* fields are NOT touched, so the buff is lost on reshuffle
        // (reset_to_base_state restores original stats) — matches Python.
        if card.has_mechanic("consume_ally") {
            let consume_idx = target_code.saturating_sub(9);
            if consume_idx < player.board.len() {
                let consumed = player.board.remove(consume_idx);
                card.attack += consumed.attack;
                card.hp += consumed.hp;
                card.max_hp += consumed.max_hp;
                player.graveyard.push(consumed);
            }
        }
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
            Some(position),
            &mut state.pending_mana_drain_by_player,
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
            None,
            &mut state.pending_mana_drain_by_player,
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
    draw_rng: &mut DrawRng,
) -> Result<(), String> {
    let (player, opponent) = state.players_for_mut(player_id)?;
    if attacker_index >= player.board.len() {
        return Err("attacker_not_found".to_string());
    }
    // Frozen attacker cannot attack (mirrors core/engine.py `_handle_attack`:
    // a frozen unit has is_ready=False after the end-turn thaw, so the
    // `is_ready` guard already blocks it; check is_frozen explicitly for
    // defense-in-depth and direct-call parity).
    if !player.board[attacker_index].is_ready || player.board[attacker_index].is_frozen {
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
            let damage_dealt =
                apply_damage(&mut opponent.hero, effective_attack, Some(attacker), draw_rng);
            attacker.is_ready = false;
            damage_dealt
        };
        if has_lifesteal && damage_dealt > 0 {
            heal_card(&mut player.hero, damage_dealt);
        }
        // instant_kill does NOT work against heroes (core/engine.py line 592):
        // Сайтама deals only base damage to the hero.
        return Ok(());
    }

    if target_code >= opponent.board.len() {
        return Err("target_not_found".to_string());
    }

    // Capture flags needed for post-exchange mechanic resolution before
    // mutable borrows scramble the board (mirrors core/engine.py `_handle_attack`).
    let target_effective_attack = compute_effective_attack(
        &opponent.board[target_code],
        &opponent.board,
        &opponent.hero,
    );
    let attacker_effective_attack =
        compute_effective_attack(&player.board[attacker_index], &player.board, &player.hero);
    let has_lifesteal = player.board[attacker_index].has_mechanic("lifesteal");
    let has_instant_kill = player.board[attacker_index].has_mechanic("instant_kill");
    let has_unit_killer = player.board[attacker_index].has_mechanic("unit_killer");
    // Cleave damage amount: parse `cleave_X` / `cleave_X_Y` → X (the _Y suffix
    // is the potion-battlecry hit count, unused for the warrior attack cleave).
    let cleave_damage = player.board[attacker_index]
        .mechanics
        .iter()
        .filter_map(|m| {
            m.strip_prefix("cleave_")
                .and_then(|rest| rest.split('_').next())
                .and_then(|chunk| chunk.parse::<i32>().ok())
        })
        .next();
    let target_had_shield = opponent.board[target_code].has_mechanic("shield");

    let damage_dealt_to_target = {
        let attacker = &mut player.board[attacker_index];
        apply_damage(
            &mut opponent.board[target_code],
            attacker_effective_attack,
            Some(attacker),
            draw_rng,
        )
    };
    let target_shield_blocked =
        target_had_shield && !opponent.board[target_code].has_mechanic("shield") && damage_dealt_to_target == 0;
    {
        let target = &mut opponent.board[target_code];
        apply_damage(
            &mut player.board[attacker_index],
            target_effective_attack,
            Some(target),
            draw_rng,
        );
    }
    // Cleave (TA-1): splash `cleave_X` damage to the neighbours of the
    // attacked board target (core/engine.py `_apply_attack_cleave`). Hits
    // only living neighbours (hp > 0). Uses apply_damage so armor/shield/
    // reflect on the neighbour are respected. Performed AFTER the damage
    // exchange, before cleanup. NO `> 0` guard: Python engine.py:~1054
    // calls `apply_damage(neighbor, damage, attacker)` unconditionally for
    // living neighbours (consuming their shield even at 0 dmg). The
    // living-neighbour check lives inside `apply_attack_cleave`, matching
    // Python's `if neighbor.hp > 0` filter. (Phase-5 latent: cleave_0
    // parity — no card uses cleave_0 today but the code must match.)
    if let Some(cleave_damage) = cleave_damage {
        apply_attack_cleave(
            opponent,
            target_code,
            cleave_damage,
            player,
            attacker_index,
            draw_rng,
        );
    }
    player.board[attacker_index].is_ready = false;
    if has_lifesteal && damage_dealt_to_target > 0 {
        heal_card(&mut player.hero, damage_dealt_to_target);
    }
    // unit_killer (TA-5): kills every attacked unit, no per-card limit
    // (core/engine.py line 640). Bypassed if the target's shield blocked
    // the damage. Sets hp=0 directly (no deathrattle trigger here —
    // deathrattle fires in cleanup_dead_units).
    if has_unit_killer {
        if !target_shield_blocked && opponent.board[target_code].hp > 0 {
            opponent.board[target_code].hp = 0;
        }
    } else if has_instant_kill {
        // instant_kill (TA-2): one-shot per card — the first attacked enemy
        // minion is set to hp=0 regardless of normal damage. Sets
        // instant_kill_used=True whether or not the kill landed (matches
        // core/engine.py line 653). Does NOT work against heroes (handled
        // above). Bypassed if shield blocked.
        let attacker = &mut player.board[attacker_index];
        if !attacker.instant_kill_used {
            attacker.instant_kill_used = true;
            if !target_shield_blocked && opponent.board[target_code].hp > 0 {
                opponent.board[target_code].hp = 0;
            }
        }
    }
    Ok(())
}

/// Apply warrior cleave splash damage to the neighbours of the attacked
/// target (core/engine.py `_apply_attack_cleave`). `target_index` is the
/// position of the primary target in `opponent.board` BEFORE cleanup; the
/// target is still on board (cleanup runs after apply_attack). Hits only
/// neighbours with hp > 0. The attacker card (at `attacker_index` in
/// `player.board`) is passed for reflect accounting.
fn apply_attack_cleave(
    opponent: &mut KernelPlayer,
    target_index: usize,
    damage: i32,
    player: &mut KernelPlayer,
    attacker_index: usize,
    draw_rng: &mut DrawRng,
) {
    // Mirror Python: iterate (target_index-1, target_index+1) and apply
    // damage to each living neighbour. The attacker-card borrow is split
    // off so reflect (which writes back to the attacker) works without
    // aliasing the opponent board borrow.
    if attacker_index >= player.board.len() {
        return;
    }
    for &neighbour_index in &[target_index.wrapping_sub(1), target_index + 1] {
        if neighbour_index >= opponent.board.len() {
            continue;
        }
        if opponent.board[neighbour_index].hp <= 0 {
            continue;
        }
        let attacker = &mut player.board[attacker_index];
        apply_damage(
            &mut opponent.board[neighbour_index],
            damage,
            Some(attacker),
            draw_rng,
        );
    }
}

fn apply_play_effects(
    _card_id: i32,
    mechanics: &[String],
    owner: &mut KernelPlayer,
    opponent: &mut KernelPlayer,
    target_code: usize,
    overdraw_to_discard: bool,
    draw_rng: &mut DrawRng,
    // Index of the just-played card in `owner.board` (warriors only), so
    // team_wide_shield can self-exclude the played card (mirrors
    // core/effects.py `effect_team_wide_shield`:
    // `u.instance_id != card.instance_id`). `None` for potions (card goes to
    // graveyard, never on board) — team_wide_shield is warrior-only anyway.
    // Also used by TAMHP Full-mode post-insertion index resolution.
    self_board_index: Option<usize>,
    // Pending-mana-drain map (`state.pending_mana_drain_by_player`). Passed
    // through so the mana_drain_X branch can schedule the overflow portion of
    // the drain into the opponent's next turn, mirroring
    // core/effects.py:540-580. The caller (apply_play_card) is responsible
    // for making the disjoint field borrow work (inline field access instead
    // of `players_for_mut`).
    pending_mana_drain_by_player: &mut BTreeMap<i32, i32>,
) {
    for mechanic in mechanics {
        if mechanic.starts_with("deathrattle_") {
            continue;
        }
        if is_random_battlecry_damage_mechanic(mechanic) {
            // card15 `battlecry_damage_X_random` (core/effects.py
            // `_apply_random_battlecry_damage`): pick a random enemy target
            // via `random.choice(list(opponent.board) + [opponent.hero])` and
            // deal X damage to it. Mechanic-driven (not card_id) so any card
            // with this mechanic behaves correctly. The random pick routes
            // through the recorded-outcome `choice_rolls` stream.
            let amount = mechanic
                .strip_prefix("battlecry_damage_")
                .and_then(|rest| rest.strip_suffix("_random"))
                .and_then(parse_i32_prefix)
                .unwrap_or(0);
            apply_random_battlecry_damage(opponent, amount, draw_rng);
        } else if let Some(amount) = mechanic
            .strip_prefix("battlecry_damage_")
            .and_then(parse_i32_prefix)
        {
            apply_damage_to_play_target(owner, opponent, target_code, amount, draw_rng);
        } else if let Some(amount) = mechanic.strip_prefix("damage_").and_then(parse_i32_prefix) {
            apply_damage_to_play_target(owner, opponent, target_code, amount, draw_rng);
        } else if let Some(amount) = mechanic
            .strip_prefix("battlecry_heal_hero_")
            .and_then(parse_i32_prefix)
        {
            owner.hero.hp = (owner.hero.hp + amount).min(owner.hero.max_hp);
        } else if mechanic == "battlecry_draw_card" {
            // No-FIFO weighted draw — mirrors core/engine.py draw_one_from_deck
            // (source="battlecry_draw_card"). Replaces the old FIFO draw.
            draw_one_from_deck(owner, overdraw_to_discard, draw_rng);
        } else if mechanic == "battlecry_freeze" || mechanic == "freeze" {
            // Freeze the targeted enemy minion (core/effects.py effect_freeze /
            // effect_battlecry_freeze). target_code 1..=7 → opponent board
            // index 0..6. A shield on the target blocks the freeze (shield
            // consumed, no freeze applied). Heroes cannot be frozen.
            if matches!(target_code, 1..=7) {
                if let Some(target) = opponent.board.get_mut(target_code - 1) {
                    if consume_shield(target) {
                        // shield absorbed the freeze
                    } else if !target.is_frozen {
                        target.is_frozen = true;
                    }
                }
            }
        } else if mechanic == "aoe_freeze" {
            // Freeze up to the first 3 enemy minions (core/effects.py
            // effect_aoe_freeze). Each unit's shield blocks the freeze for
            // that unit (shield consumed) — `consume_shield` then skip.
            for target in opponent.board.iter_mut().take(3) {
                if consume_shield(target) {
                    continue;
                }
                if !target.is_frozen {
                    target.is_frozen = true;
                }
            }
        } else if mechanic == "desk_freeze" {
            // Freeze ALL enemy minions (core/effects.py effect_desk_freeze).
            for target in opponent.board.iter_mut() {
                if consume_shield(target) {
                    continue;
                }
                if !target.is_frozen {
                    target.is_frozen = true;
                }
            }
        } else if let Some(rest) = mechanic.strip_prefix("freeze_") {
            // `freeze_X` (dynamic) — freeze the targeted enemy minion. Same
            // target mapping as `freeze`/`battlecry_freeze`. The `_X` count
            // is parsed only to distinguish the mechanic from `freeze_on_play`
            // (which does NOT match — `on_play` is non-numeric); the count is
            // otherwise unused for the single-target freeze.
            if rest.split('_').next().and_then(|c| c.parse::<i32>().ok()).is_some() {
                if matches!(target_code, 1..=7) {
                    if let Some(target) = opponent.board.get_mut(target_code - 1) {
                        if consume_shield(target) {
                            // shield absorbed
                        } else if !target.is_frozen {
                            target.is_frozen = true;
                        }
                    }
                }
            }
        } else if mechanic == "aoe_silence" {
            // AOE-SILENCE-1 (card 47 Солдатик): strip ALL mechanics from up to
            // 3 enemy minions that currently have mechanics. Mirrors
            // core/effects.py `effect_aoe_silence`: candidates are
            // `[u for u in opponent.board if u.mechanics]` and `_silence_units`
            // clears `unit.mechanics = []` (limit=3). Status flags
            // (is_frozen, is_ready, instant_kill_used, ...) are NOT touched —
            // only the mechanics list. No damage; shield does NOT protect
            // (silence strips the shield mechanic too, since shield lives in
            // `mechanics`). The just-played card is irrelevant here (targets
            // are enemy minions).
            let mut silenced = 0;
            for target in opponent.board.iter_mut() {
                if target.mechanics.is_empty() {
                    continue;
                }
                target.mechanics.clear();
                silenced += 1;
                if silenced >= 3 {
                    break;
                }
            }
        } else if mechanic == "team_wide_shield" {
            // TWS-1/2 (card 48 Соул Гудман): grant a one-shot `shield` to up
            // to 3 friendly minions, EXCLUDING the just-played card itself.
            // Mirrors core/effects.py `effect_team_wide_shield`:
            // `targets = [u for u in owner.board if u.instance_id !=
            //  card.instance_id]` then `_grant_shields(targets, limit=3)`.
            // At this point the played card is already inserted into
            // owner.board at `self_board_index`; self-exclusion skips that
            // index so it does NOT consume one of the 3 shield slots. Units
            // that already have `shield` are skipped (don't waste the limit).
            // The hero is NOT a target (mechanic is "cards on board" only).
            let mut granted = 0;
            for (idx, unit) in owner.board.iter_mut().enumerate() {
                if Some(idx) == self_board_index {
                    continue;
                }
                if unit.has_mechanic("shield") {
                    continue;
                }
                unit.mechanics.push("shield".to_string());
                granted += 1;
                if granted >= 3 {
                    break;
                }
            }
        } else if let Some(amount) = mechanic
            .strip_prefix("target_ally_max_hp_plus_universal_")
            .and_then(parse_i32_prefix)
        {
            // TAMHP-1/2/3 (card 52 Криста Ленц, universal variant): bump the
            // targeted friendly unit's (or own hero's) max_hp by `amount` via
            // DIRECT increase. Current hp is NOT changed (no healing; the
            // delta only raises the ceiling). NO clamp via heal_card (audit
            // risk note). Mirrors core/effects.py
            // `target_ally_max_hp_plus_universal_N` handler: `unit.max_hp +=
            // hp_bonus` / `owner.hero.max_hp += hp_bonus`. target_code 9..=15
            // → owner.board[0..6]; target_code 16 → owner.hero. Universal
            // allows hero; the non-universal variant below forbids hero.
            apply_max_hp_plus(owner, target_code, amount, true, self_board_index);
        } else if let Some(rest) = mechanic.strip_prefix("target_ally_max_hp_plus_") {
            // TAMHP non-universal variant (`target_ally_max_hp_plus_N`,
            // allow_hero=False). The universal branch above handles the
            // `_universal_` prefix, so `rest` here is the bare `N` (the
            // universal case already returned). Guard: only parse if `rest`
            // is a pure integer (skip `universal_...` which was already
            // consumed above — defensive double-check).
            if let Some(amount) = parse_i32_prefix(rest) {
                apply_max_hp_plus(owner, target_code, amount, false, self_board_index);
            }
        } else if let Some(amount) = mechanic.strip_prefix("mana_drain_").and_then(parse_i32_prefix)
        {
            // Mana-drain (card 12 Кража Маны, `mana_drain_X`). Two-stage, mirrors
            // core/effects.py:540-580 `make_drain_handler`:
            //   1. Immediate drain: `current_drained = min(opponent.mana, amount)`;
            //      `opponent.mana -= current_drained`.
            //   2. Schedule the overflow into the opponent's NEXT turn:
            //      `existing_pending = pending.get(opponent.user_id, 0)`;
            //      `future_pool = max(0, opponent.max_mana - existing_pending)`;
            //      `future_drained = min(amount - current_drained, future_pool)`;
            //      if > 0, `pending[opponent.user_id] = existing_pending +
            //      future_drained`.
            //   3. Owner receives the TOTAL drained:
            //      `owner.mana = min(owner.max_mana, owner.mana + current_drained
            //      + future_drained)`.
            // The pending half is applied in `apply_end_turn` after
            // `opponent.mana = opponent.max_mana` (core/engine.py:700-703).
            let current_drained = opponent.mana.min(amount);
            opponent.mana -= current_drained;
            let opponent_user_id = opponent.user_id;
            let existing_pending = pending_mana_drain_by_player
                .get(&opponent_user_id)
                .copied()
                .unwrap_or(0);
            let future_pool = (opponent.max_mana - existing_pending).max(0);
            let future_drained = (amount - current_drained).min(future_pool).max(0);
            if future_drained > 0 {
                pending_mana_drain_by_player
                    .insert(opponent_user_id, existing_pending + future_drained);
            }
            let drained = current_drained + future_drained;
            owner.mana = (owner.mana + drained).min(owner.max_mana);
        } else if let Some(amount) = mechanic
            .split("aoe_damage_")
            .nth(1)
            .and_then(|rest| rest.split('_').next())
            .and_then(parse_i32_prefix)
        {
            // AOE damage (card 10 potion "Импульс Бездны" `aoe_damage_2`, plus
            // the `battlecry_aoe_damage_X` / `spell_aoe_damage_X` variants).
            // Mirrors core/effects.py `_register_aoe_damage_effects` +
            // `effect_battlecry_aoe_damage_*` / `effect_spell_aoe_damage_2`:
            // `for unit in opponent.board: apply_damage(unit, dmg)` — ALL
            // enemy minions, NOT the enemy hero. Uses apply_damage (goes
            // through shield/armor/reflect modifiers, attacker=None so no
            // reflect). `deathrattle_aoe_damage_X` is skipped at the top of
            // the loop. No target needed (mask: no-target slot).
            for target in opponent.board.iter_mut() {
                apply_damage(target, amount, None, draw_rng);
            }
        } else if let Some(heal_amount) = mechanic
            .strip_prefix("battlecry_heal_target_")
            .or_else(|| mechanic.strip_prefix("spell_heal_target_"))
            .or_else(|| mechanic.strip_prefix("heal_target_"))
            .and_then(parse_i32_prefix)
        {
            // heal_target_X / battlecry_heal_target_X / spell_heal_target_X
            // (card 36 Юни `battlecry_heal_target_3`, card 35 Фрирер
            // `battlecry_heal_target_5`). Mirrors core/effects.py
            // `_register_battlecry_heal_target_effects` + the dynamic regex
            // at effects.py:1441-1461: heal the targeted FRIENDLY unit (by
            // instance_id) or the owner hero. apply_heal clamps to max_hp.
            // target_code 9..=15 → friendly board (post-insertion index via
            // self_board_index, same indexing as TAMHP); 16 → own hero.
            apply_heal_to_play_target(owner, target_code, heal_amount, self_board_index);
        } else if mechanic == "delete_target" {
            // delete_target (card 13 potion "Черная Дыра"). Mirrors
            // core/effects.py `effect_delete_target`: find the enemy unit by
            // instance_id; if consume_shield(unit) → shield BLOCKS the delete
            // AND is consumed (return, no remove); else remove from
            // opponent.board → push to opponent.graveyard, NO deathrattle
            // fires (remove, not kill). Targets enemy minions ONLY
            // (target_code 1..=7); hero target (8) is a no-op (Python loops
            // opponent.board, not hero).
            if matches!(target_code, 1..=7) {
                let idx = target_code - 1;
                if idx < opponent.board.len() {
                    if consume_shield(&mut opponent.board[idx]) {
                        // shield consumed, delete blocked
                    } else {
                        let unit = opponent.board.remove(idx);
                        opponent.graveyard.push(unit);
                    }
                }
            }
        } else if mechanic == "choose_shield_damage" {
            // choose_shield_damage (card 21 warrior "Геральт"). Mirrors
            // core/effects.py `effect_choose_shield_damage`: if target_id →
            // apply 3 damage to the enemy unit (or enemy hero); else (no
            // target) → grant `shield` to the PLAYED CARD itself (the warrior
            // on board). The mask exposes both the no-target slot (base+0)
            // and enemy target slots (1..=8). target_code 0 → shield branch.
            if target_code == 0 {
                if let Some(pos) = self_board_index {
                    if let Some(card) = owner.board.get_mut(pos) {
                        if !card.has_mechanic("shield") {
                            card.mechanics.push("shield".to_string());
                        }
                    }
                }
            } else {
                match target_code {
                    1..=7 => {
                        if let Some(target) = opponent.board.get_mut(target_code - 1) {
                            apply_damage(target, 3, None, draw_rng);
                        }
                    }
                    8 => {
                        apply_damage(&mut opponent.hero, 3, None, draw_rng);
                    }
                    _ => {}
                }
            }
        } else if mechanic == "cast_random_spell" {
            // cast_random_spell (card 26 warrior "Мидория"). Mirrors
            // core/effects.py `effect_cast_random_spell` (effects.py:838-951).
            // RNG protocol: spell_choice = random.randint(1,4) →
            // randint_rolls; spell 1 target = random.choice(opponent.board +
            // [hero]) → choice_rolls; spell 3 freeze targets =
            // random.sample(unfrozen_enemies, k) → sample_rolls. Scaling uses
            // card.level (level=1 in the classic training env → dmg=4,
            // heal=5, freeze_count=1, buff=2); mirrored for any level.
            let level = match self_board_index {
                Some(pos) => owner.board.get(pos).map(|c| c.level).unwrap_or(1),
                None => 1,
            };
            let spell_choice = roll_range(1, 4, draw_rng);
            if spell_choice == 1 {
                // Texas Smash: dmg = 4 + (level-1) to a random enemy target
                // (opponent.board + [opponent.hero]).
                let dmg = 4 + (level - 1);
                let n = opponent.board.len() + 1;
                let idx = roll_choice(n, draw_rng);
                if idx < opponent.board.len() {
                    let target = &mut opponent.board[idx];
                    apply_damage(target, dmg, None, draw_rng);
                } else {
                    apply_damage(&mut opponent.hero, dmg, None, draw_rng);
                }
            } else if spell_choice == 2 {
                // Recovery: heal owner hero by 5 + (level-1) (clamped to max_hp).
                let heal = 5 + (level - 1);
                owner.hero.hp = (owner.hero.hp + heal).min(owner.hero.max_hp);
            } else if spell_choice == 3 {
                // Blackwhip: freeze up to freeze_count random unfrozen enemies.
                // `freeze_count = 2 if level >= 5 else 1`.
                let freeze_count = if level >= 5 { 2 } else { 1 };
                let unfrozen_indices: Vec<usize> = opponent
                    .board
                    .iter()
                    .enumerate()
                    .filter_map(|(i, u)| if !u.is_frozen { Some(i) } else { None })
                    .collect();
                let k = freeze_count.min(unfrozen_indices.len());
                let selected = roll_sample(unfrozen_indices.len(), k, draw_rng);
                for sel in selected {
                    let board_idx = unfrozen_indices[sel];
                    if consume_shield(&mut opponent.board[board_idx]) {
                        continue;
                    }
                    if !opponent.board[board_idx].is_frozen {
                        opponent.board[board_idx].is_frozen = true;
                    }
                }
            } else {
                // spell_choice == 4: Full Cowl — buff the played card itself.
                // `buff = 2 + ((level-1)//2)` to attack/hp/max_hp.
                let buff = 2 + (level - 1) / 2;
                if let Some(pos) = self_board_index {
                    if let Some(card) = owner.board.get_mut(pos) {
                        card.attack += buff;
                        card.hp += buff;
                        card.max_hp += buff;
                    }
                }
            }
        }
    }
}

/// Apply `heal_target_X` / `battlecry_heal_target_X` / `spell_heal_target_X`
/// to the play target (card 36 Юни, card 35 Фрирер). Mirrors core/effects.py
/// `_register_battlecry_heal_target_effects` + the dynamic regex at
/// effects.py:1441-1461: `apply_heal(unit, heal)` which clamps
/// `hp = min(max_hp, hp + heal)`. target_code 9..=15 → friendly board minion,
/// 16 → own hero. Uses the SAME Full-mode post-insertion indexing as
/// `apply_max_hp_plus` (the target is resolved to a PRE-insertion board
/// index at decode time; when the new warrior is inserted at `pos <=
/// pre_index` the target shifts right by 1). `self_board_index=None` for
/// potions → no shift.
fn apply_heal_to_play_target(
    owner: &mut KernelPlayer,
    target_code: usize,
    amount: i32,
    self_board_index: Option<usize>,
) {
    match target_code {
        9..=15 => {
            let pre_index = target_code - 9;
            let post_index = match self_board_index {
                Some(pos) if pos <= pre_index => pre_index + 1,
                _ => pre_index,
            };
            if let Some(target) = owner.board.get_mut(post_index) {
                target.hp = (target.hp + amount).min(target.max_hp);
            }
        }
        16 => {
            owner.hero.hp = (owner.hero.hp + amount).min(owner.hero.max_hp);
        }
        _ => {}
    }
}

/// Apply `target_ally_max_hp_plus[_universal]_N` to the play target. Bumps
/// `max_hp` by `amount` directly (no hp change, no heal_card clamp). When
/// `allow_hero` is false, target_code 16 (own hero) is ignored — matching the
/// non-universal variant in core/effects.py (`allow_hero=False`).
///
/// Full-mode post-insertion indexing (Phase-5 latent: TAMHP Full-mode
/// indexing). The play target is resolved to a PRE-insertion board index
/// (`target_code - 9`) at decode time — Python resolves it to a stable
/// `instance_id` (classic_actions_v1.py:84,122-123) and searches by
/// `instance_id` post-effect (core/effects.py:1126-1133). Rust KernelCard has
/// no instance_id, so we instead compute the POST-insertion index from the
/// PRE-insertion index and the insert position (`self_board_index`): when the
/// new warrior is inserted at `pos <= pre_index`, every board card at
/// `pre_index` shifts right by 1, so the post-insertion index is
/// `pre_index + 1`; otherwise the target stays at `pre_index`. `consume_ally`
/// consumes its target BEFORE insert (separate path) so it does not shift the
/// TAMHP target. Default append_only places at `board.len()` (pos > pre_index
/// always) so it is unaffected; Full mode is now correct.
fn apply_max_hp_plus(
    owner: &mut KernelPlayer,
    target_code: usize,
    amount: i32,
    allow_hero: bool,
    self_board_index: Option<usize>,
) {
    match target_code {
        9..=15 => {
            let pre_index = target_code - 9;
            let post_index = match self_board_index {
                Some(pos) if pos <= pre_index => pre_index + 1,
                _ => pre_index,
            };
            if let Some(target) = owner.board.get_mut(post_index) {
                target.max_hp += amount;
            }
        }
        16 if allow_hero => {
            owner.hero.max_hp += amount;
        }
        _ => {}
    }
}

fn apply_damage_to_play_target(
    owner: &mut KernelPlayer,
    opponent: &mut KernelPlayer,
    target_code: usize,
    amount: i32,
    draw_rng: &mut DrawRng,
) {
    match target_code {
        1..=7 => {
            if let Some(target) = opponent.board.get_mut(target_code - 1) {
                apply_damage(target, amount, None, draw_rng);
            }
        }
        8 => {
            apply_damage(&mut opponent.hero, amount, None, draw_rng);
        }
        16 => {
            apply_damage(&mut owner.hero, amount, None, draw_rng);
        }
        9..=15 => {
            if let Some(target) = owner.board.get_mut(target_code - 9) {
                apply_damage(target, amount, None, draw_rng);
            }
        }
        _ => {}
    }
}

/// Card15 `battlecry_damage_X_random` (core/effects.py
/// `_apply_random_battlecry_damage`): build `targets = list(opponent.board) +
/// [opponent.hero]`, pick one via `random.choice`, and deal `amount` damage.
/// The random pick routes through `roll_choice` (recorded-outcome
/// `choice_rolls` stream) so Rust reproduces Python's recorded target in
/// fixtures and rolls fresh in training. Shield/armor on the target apply via
/// `apply_damage` (mirrors Python `apply_damage` → `apply_damage_modifiers`).
fn apply_random_battlecry_damage(
    opponent: &mut KernelPlayer,
    amount: i32,
    draw_rng: &mut DrawRng,
) {
    // NO `amount <= 0` early-return: mirrors Python `_apply_random_battlecry_damage`
    // (core/effects.py:67-83), which always builds `targets = opponent.board +
    // [opponent.hero]`, calls `random.choice` (consuming the choice_rolls RNG
    // stream), and `apply_damage(target, amount)` (which calls consume_shield on
    // a shielded target even at amount=0). No catalog card produces amount=0
    // (regex `battlecry_damage_(\d+)_random` requires \d+), so this is a latent
    // forward-compat parity fix, not a behavior change for current cards.
    let n = opponent.board.len() + 1; // +1 for hero
    let idx = roll_choice(n, draw_rng);
    if idx < opponent.board.len() {
        // Order matters: borrow the board target mutably without also
        // borrowing `opponent.hero` (split borrow).
        let target = &mut opponent.board[idx];
        apply_damage(target, amount, None, draw_rng);
    } else {
        apply_damage(&mut opponent.hero, amount, None, draw_rng);
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
        // Mirrors ai/train_v2/classic_actions_v1.py:589
        // `(att_pos + 1) / (_NUM_BOARD + 1)` where _NUM_BOARD=5 (line 50)
        // → divisor 6.0. This is the GAME-RULE board cap+1 (GAME_BOARD_CAP=5),
        // NOT the codec NUM_BOARD(7)+1=8. Phase-2: gap AC-FFI-1 (att_pos divisor).
        .map(|p| ((p as f64 + 1.0) / (GAME_BOARD_CAP as f64 + 1.0)) as f32)
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
    // 20-slot region: 2 heroes + 5 own board + 5 enemy board + 4 own hand
    // + 4 zero padding (matches classic_obs_v1._encode_card_slots).
    write_shape(out, offset, Some(&me.hero), None, None, None);
    write_shape(out, offset, Some(&enemy.hero), None, None, None);
    for i in 0..5 {
        write_shape(out, offset, me.board.get(i), Some(i), None, None);
    }
    for i in 0..5 {
        write_shape(out, offset, enemy.board.get(i), Some(i), None, None);
    }
    for i in 0..4 {
        write_shape(out, offset, me.hand.get(i), None, Some(i), None);
    }
    // `out` is zero-initialized by encode_observation_v1, so padding is just
    // advancing the offset by 4 empty card-shape slots.
    *offset += 4 * CARD_SHAPE_DIM;
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
                // TAMHP (card 52): target_ally_max_hp_plus[_universal]_N
                // requires a friendly-minion (universal: +hero) target. The
                // universal check must come first — `target_ally_max_hp_plus_`
                // would strip to `universal_N` which fails the int parse, so
                // both prefixes are listed.
                || has_prefixed_number(m, "target_ally_max_hp_plus_universal_")
                || has_prefixed_number(m, "target_ally_max_hp_plus_")
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
    // Mirrors core/effects.py `is_random_battlecry_damage_card`: a card opts
    // into random battlecry damage via a `battlecry_damage_X_random` mechanic
    // (e.g. card15 "Тока Киришима" carries `battlecry_damage_1_random`).
    // Mechanic-driven so any future card with the same mechanic behaves
    // correctly — no card_id hardcode.
    card.mechanics
        .iter()
        .any(|m| is_random_battlecry_damage_mechanic(m))
}

/// Matches the regex `battlecry_damage_\d+_random$` (core/effects.py line 62).
fn is_random_battlecry_damage_mechanic(mechanic: &str) -> bool {
    mechanic
        .strip_prefix("battlecry_damage_")
        .and_then(|rest| rest.strip_suffix("_random"))
        .map(|num| !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit()))
        .unwrap_or(false)
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

fn apply_damage(
    target: &mut KernelCard,
    damage: i32,
    attacker: Option<&mut KernelCard>,
    draw_rng: &mut DrawRng,
) -> i32 {
    let modified_damage = apply_damage_modifiers(target, damage, draw_rng);
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
                let reflect_damage = apply_damage_modifiers(attacker, reflect_amount, draw_rng);
                attacker.hp = (attacker.hp - reflect_damage).max(0);
            }
        }
    }

    actual_damage
}

fn apply_damage_modifiers(target: &mut KernelCard, damage: i32, draw_rng: &mut DrawRng) -> i32 {
    if target.has_mechanic("permanent_shield") {
        return 0;
    }
    if consume_shield(target) {
        return 0;
    }
    // Armor (TA-4): `armor_X` is a fixed reduction; `armor_X_Y` rolls
    // `random.randint(X, Y)` inclusive (core/effects.py line 1219-1223).
    // Only the FIRST armor mechanic on the card is applied (Python `break`).
    // The armor mechanic is NOT consumed — it persists across attacks.
    let mut armor_value = 0;
    for mechanic in &target.mechanics {
        if let Some(rest) = mechanic.strip_prefix("armor_") {
            if let Some((min_str, max_str)) = rest.split_once('_') {
                if let (Ok(min_val), Ok(max_val)) = (min_str.parse::<i32>(), max_str.parse::<i32>())
                {
                    // Range roll — deterministic via recorded-outcome RNG.
                    armor_value = roll_range(min_val, max_val, draw_rng);
                }
            } else if let Ok(fixed) = rest.parse::<i32>() {
                armor_value = fixed;
            }
            break;
        }
    }
    (damage - armor_value).max(0)
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
    card.instant_kill_used = false;
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

fn cleanup_dead_units(state: &mut KernelState, draw_rng: &mut DrawRng) {
    loop {
        let before_p1 = state.p1.board.len();
        let before_p2 = state.p2.board.len();
        {
            let (p1, p2) = (&mut state.p1, &mut state.p2);
            cleanup_dead_units_for_player(p1, p2, draw_rng);
            cleanup_dead_units_for_player(p2, p1, draw_rng);
        }
        if state.p1.board.len() == before_p1 && state.p2.board.len() == before_p2 {
            break;
        }
    }
}

fn cleanup_dead_units_for_player(
    player: &mut KernelPlayer,
    opponent: &mut KernelPlayer,
    draw_rng: &mut DrawRng,
) {
    // --- Rebirth pre-pass (Phase 3: REBIRTH-1). Mirrors
    // core/engine.py::_cleanup_dead_units: a unit with rebirth_N at lethal
    // damage survives with N HP (one-shot — the rebirth_N mechanic string is
    // removed). Runs BEFORE deathrattle/CAP so the saved unit is not
    // considered dead, its deathrattle does not fire. This pre-pass runs EACH
    // cleanup loop iteration (the outer cleanup_dead_units loop repeats until
    // board sizes stabilise, because deathrattles can kill more units — those
    // subsequent deaths must also consider rebirth).
    for unit in player.board.iter_mut() {
        if unit.hp > 0 {
            continue;
        }
        if let Some(rebirth_hp) = parse_rebirth(&unit.mechanics) {
            if rebirth_hp > 0 {
                unit.hp = rebirth_hp;
                consume_rebirth(&mut unit.mechanics);
            }
        }
    }

    // --- crime_and_punishment (Phase 3: CAP-1). Mirrors
    // core/engine.py::_cleanup_dead_units: parse the hero passive once, then
    // for EACH dead friendly minion deal N damage to the OPPONENT hero via
    // DIRECT hp subtraction (max(0, hp - N)) — NOT apply_damage. This bypasses
    // armor/reflect/lifesteal and does NOT chain deathrattles from this damage
    // (audit risk note: confirmed in Python — line `opponent.hero.hp = max(0,
    // opponent.hero.hp - cap_damage)`).
    let cap_damage = parse_crime_and_punishment(&player.hero.mechanics);

    let mut alive = Vec::with_capacity(player.board.len());
    for unit in player.board.drain(..) {
        if unit.hp <= 0 {
            apply_deathrattle_effects(&unit, opponent, draw_rng);
            if cap_damage > 0 && opponent.hero.hp > 0 {
                opponent.hero.hp = (opponent.hero.hp - cap_damage).max(0);
            }
            player.graveyard.push(unit);
        } else {
            alive.push(unit);
        }
    }
    player.board = alive;
}

/// Parse `rebirth_N` from a mechanics list → N (first match). Mirrors
/// core/effects.py::parse_rebirth.
fn parse_rebirth(mechanics: &[String]) -> Option<i32> {
    for m in mechanics {
        if let Some(rest) = m.strip_prefix("rebirth_") {
            if let Ok(n) = rest.parse::<i32>() {
                return Some(n);
            }
        }
    }
    None
}

/// Remove all `rebirth_*` mechanic strings (one-shot consumption). Mirrors
/// core/effects.py::consume_rebirth.
fn consume_rebirth(mechanics: &mut Vec<String>) {
    mechanics.retain(|m| !m.starts_with("rebirth_"));
}

/// Parse `crime_and_punishment_N` from a mechanics list → N (first match).
/// Mirrors core/effects.py::parse_crime_and_punishment.
fn parse_crime_and_punishment(mechanics: &[String]) -> i32 {
    for m in mechanics {
        if let Some(rest) = m.strip_prefix("crime_and_punishment_") {
            if let Ok(n) = rest.parse::<i32>() {
                return n;
            }
        }
    }
    0
}

fn apply_deathrattle_effects(
    unit: &KernelCard,
    opponent: &mut KernelPlayer,
    draw_rng: &mut DrawRng,
) {
    for mechanic in &unit.mechanics {
        let Some(amount) = mechanic
            .strip_prefix("deathrattle_aoe_damage_")
            .and_then(parse_i32_prefix)
        else {
            continue;
        };
        for target in &mut opponent.board {
            if target.hp > 0 {
                apply_damage(target, amount, None, draw_rng);
            }
        }
        if opponent.hero.hp > 0 {
            apply_damage(&mut opponent.hero, amount, None, draw_rng);
        }
        break;
    }
}

fn check_game_over(state: &mut KernelState) {
    // Mirrors core/engine.py:932-941 `_check_game_over`: only the three death
    // branches assign status; there is NO else-branch. An already-terminal
    // state (draw/p1_win/p2_win) is left UNCHANGED when nobody is newly dead,
    // so Rust never resurrects a terminal state. (Phase-5 latent: status
    // overwrite — removed the unconditional `else { ongoing }`.)
    let p1_dead = state.p1.hero.hp <= 0;
    let p2_dead = state.p2.hero.hp <= 0;
    if p1_dead && p2_dead {
        state.status = "draw".to_string();
    } else if p1_dead {
        state.status = "p2_win".to_string();
    } else if p2_dead {
        state.status = "p1_win".to_string();
    }
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

    // ---- Phase 3: rebirth (REBIRTH-1) ----

    fn card_with_mechanics(id: i32, cost: i32, atk: i32, hp: i32, mechanics: Vec<&str>) -> KernelCard {
        KernelCard {
            card_id: id,
            card_type: "warrior".to_string(),
            mana_cost: cost,
            attack: atk,
            hp,
            max_hp: hp,
            mechanics: mechanics.into_iter().map(String::from).collect(),
            is_ready: false,
            is_frozen: false,
            level: 1,
            ..Default::default()
        }
    }

    /// Test helper: run `cleanup_dead_units_for_player` with a live seeded
    /// DrawRng (the rebirth/CAP tests don't trigger randint rolls — armor
    /// is bypassed by CAP's direct subtraction and no deathrattle damage
    /// hits an armor_X_Y target here — so any seed suffices).
    fn cleanup_for_test(p: &mut KernelPlayer, opp: &mut KernelPlayer) {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        cleanup_dead_units_for_player(p, opp, &mut draw_rng);
    }

    #[test]
    fn parse_rebirth_extracts_n_from_mechanic() {
        assert_eq!(parse_rebirth(&["rebirth_1".to_string()]), Some(1));
        assert_eq!(parse_rebirth(&["rebirth_3".to_string()]), Some(3));
        assert_eq!(parse_rebirth(&["shield".to_string(), "rebirth_2".to_string()]), Some(2));
        assert_eq!(parse_rebirth(&["shield".to_string()]), None);
        assert_eq!(parse_rebirth(&[]), None);
    }

    #[test]
    fn consume_rebirth_removes_all_rebirth_mechanics() {
        let mut m = vec!["rebirth_1".to_string(), "shield".to_string(), "rebirth_2".to_string()];
        consume_rebirth(&mut m);
        assert_eq!(m, vec!["shield".to_string()]);
    }

    #[test]
    fn cleanup_rebirth_resurrects_unit_with_n_hp_and_consumes_charge() {
        // Card 50 Бан: rebirth_1. Unit at 0 hp survives with 1 hp, mechanic removed.
        let mut p = player_with(Vec::new(), Vec::new());
        p.board = vec![card_with_mechanics(50, 3, 2, 0, vec!["rebirth_1"])];
        let mut opp = player_with(Vec::new(), Vec::new());
        cleanup_for_test(&mut p, &mut opp);
        assert_eq!(p.board.len(), 1, "rebirthed unit stays on board");
        assert_eq!(p.board[0].hp, 1, "hp set to rebirth_N");
        assert!(!p.board[0].mechanics.iter().any(|m| m.starts_with("rebirth_")), "charge consumed");
        assert_eq!(p.graveyard.len(), 0, "not sent to graveyard");
    }

    #[test]
    fn cleanup_rebirth_does_not_trigger_deathrattle_for_saved_unit() {
        // A unit with both rebirth_1 and deathrattle_aoe_damage_2: rebirth saves it,
        // deathrattle must NOT fire (opponent hero hp unchanged).
        let mut p = player_with(Vec::new(), Vec::new());
        p.board = vec![card_with_mechanics(50, 3, 2, 0, vec!["rebirth_1", "deathrattle_aoe_damage_2"])];
        let mut opp = player_with(Vec::new(), Vec::new());
        let hero_hp_before = opp.hero.hp;
        cleanup_for_test(&mut p, &mut opp);
        assert_eq!(p.board.len(), 1);
        assert_eq!(p.board[0].hp, 1);
        assert_eq!(opp.hero.hp, hero_hp_before, "deathrattle did NOT fire for rebirthed unit");
    }

    #[test]
    fn cleanup_without_rebirth_sends_unit_to_graveyard() {
        let mut p = player_with(Vec::new(), Vec::new());
        p.board = vec![card_with_mechanics(27, 2, 1, 0, vec![])];
        let mut opp = player_with(Vec::new(), Vec::new());
        cleanup_for_test(&mut p, &mut opp);
        assert_eq!(p.board.len(), 0);
        assert_eq!(p.graveyard.len(), 1);
        assert_eq!(p.graveyard[0].card_id, 27);
    }

    // ---- Phase 3: crime_and_punishment (CAP-1) ----

    #[test]
    fn parse_crime_and_punishment_extracts_n() {
        assert_eq!(parse_crime_and_punishment(&["crime_and_punishment_2".to_string()]), 2);
        assert_eq!(parse_crime_and_punishment(&["armor_1".to_string(), "crime_and_punishment_3".to_string()]), 3);
        assert_eq!(parse_crime_and_punishment(&["armor_1".to_string()]), 0);
        assert_eq!(parse_crime_and_punishment(&[]), 0);
    }

    #[test]
    fn cleanup_cap_deals_direct_hp_subtraction_to_opponent_hero() {
        // Card 49 Достоевский hero: crime_and_punishment_2. When a friendly minion
        // dies, opponent hero takes 2 damage via DIRECT subtraction (not apply_damage).
        let mut p = player_with(Vec::new(), Vec::new());
        p.hero.mechanics = vec!["crime_and_punishment_2".to_string()];
        p.board = vec![card_with_mechanics(27, 2, 1, 0, vec![])];
        let mut opp = player_with(Vec::new(), Vec::new());
        // Give opponent hero armor to confirm CAP bypasses it.
        opp.hero.mechanics = vec!["armor_1_3".to_string()];
        let opp_hero_hp_before = opp.hero.hp;
        cleanup_for_test(&mut p, &mut opp);
        assert_eq!(p.board.len(), 0, "dead unit removed");
        assert_eq!(p.graveyard.len(), 1);
        // Direct subtraction: hp - 2, armor NOT applied.
        assert_eq!(opp.hero.hp, opp_hero_hp_before - 2, "CAP bypasses armor (direct subtraction)");
    }

    #[test]
    fn cleanup_cap_fires_for_each_dead_friendly_minion() {
        let mut p = player_with(Vec::new(), Vec::new());
        p.hero.mechanics = vec!["crime_and_punishment_2".to_string()];
        p.board = vec![
            card_with_mechanics(27, 2, 1, 0, vec![]),
            card_with_mechanics(28, 2, 1, 0, vec![]),
        ];
        let mut opp = player_with(Vec::new(), Vec::new());
        let opp_hero_hp_before = opp.hero.hp;
        cleanup_for_test(&mut p, &mut opp);
        assert_eq!(p.graveyard.len(), 2);
        assert_eq!(opp.hero.hp, opp_hero_hp_before - 4, "2 dead minions → 2*2=4 CAP damage");
    }

    #[test]
    fn cleanup_cap_does_not_fire_when_no_friendly_minion_dies() {
        let mut p = player_with(Vec::new(), Vec::new());
        p.hero.mechanics = vec!["crime_and_punishment_2".to_string()];
        p.board = vec![card_with_mechanics(27, 2, 1, 3, vec![])];
        let mut opp = player_with(Vec::new(), Vec::new());
        let opp_hero_hp_before = opp.hero.hp;
        cleanup_for_test(&mut p, &mut opp);
        assert_eq!(p.board.len(), 1, "alive unit stays");
        assert_eq!(opp.hero.hp, opp_hero_hp_before, "no CAP when no minion dies");
    }

    // ---- Phase 3: consume_ally (CLN-3) ----

    #[test]
    fn cleanup_rebirth_pre_pass_runs_each_loop_iteration() {
        // Two units die in the same cleanup pass: one with rebirth, one without.
        // Both should be handled correctly in a single call.
        let mut p = player_with(Vec::new(), Vec::new());
        p.board = vec![
            card_with_mechanics(50, 3, 2, 0, vec!["rebirth_1"]),
            card_with_mechanics(27, 2, 1, 0, vec![]),
        ];
        let mut opp = player_with(Vec::new(), Vec::new());
        cleanup_for_test(&mut p, &mut opp);
        assert_eq!(p.board.len(), 1, "rebirthed unit stays, dead unit removed");
        assert_eq!(p.board[0].card_id, 50);
        assert_eq!(p.board[0].hp, 1);
        assert_eq!(p.graveyard.len(), 1);
        assert_eq!(p.graveyard[0].card_id, 27);
    }

    #[test]
    fn apply_play_card_consume_ally_removes_consumed_and_adds_stats() {
        // Card 20 Канеки Кен: consume_ally. Consume a friendly 2/3 ally →
        // card becomes (1+2)/(2+3) = 3/5, ally removed to graveyard.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(20, 3, 1, 2, vec!["consume_ally"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p1.board = vec![card_with_mechanics(27, 2, 2, 3, vec![])];
        state.current_turn_owner_id = 1;
        state.status = "ongoing".to_string();
        let kernel = RolloutKernel::new(KernelConfig::default());
        // Find a legal action_id for the consume_ally play (target_code 9 =
        // friendly board[0]). build_action_mask encodes play targets at
        // base + 9 + board_idx; with AppendOnly only position=board.len() is
        // legal.
        let mask = build_action_mask(&state, 1, PlacementMode::AppendOnly);
        // Find the consume_ally play action: PlayCard with hand_index=0 and
        // target_code=9 (friendly board[0]). EndTurn at id 0 is also legal,
        // so scan for the right decode.
        let action_id = (0..mask.len())
            .filter(|&i| mask[i] == 1.0)
            .find(|&i| matches!(
                decode_action_id(i),
                Some(CandidateAction::PlayCard { hand_index: 0, target_code: 9, .. })
            ))
            .expect("consume_ally play action is legal");
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng);
        assert!(out.is_ok(), "consume_ally play applies: {:?}", out.err());
        let new = out.unwrap().state;
        assert_eq!(new.p1.board.len(), 1, "consumed ally removed, new card placed");
        assert_eq!(new.p1.board[0].card_id, 20);
        assert_eq!(new.p1.board[0].attack, 3, "1 + 2 = 3");
        assert_eq!(new.p1.board[0].hp, 5, "2 + 3 = 5");
        assert_eq!(new.p1.board[0].max_hp, 5, "2 + 3 = 5");
        assert_eq!(new.p1.graveyard.len(), 1, "consumed ally in graveyard");
        assert_eq!(new.p1.graveyard[0].card_id, 27);
    }

    // ---- Phase 5: aoe_silence / team_wide_shield / target_ally_max_hp_plus ----

    /// Find a legal PlayCard action for `hand_index` with `target_code`.
    /// Mirrors find_attack_action. AppendOnly means only position=board.len()
    /// is legal for warriors, so the scan finds the single matching action.
    fn find_play_action(state: &KernelState, player_id: i32, hand_index: usize, target_code: usize) -> usize {
        let mask = build_action_mask(state, player_id, PlacementMode::AppendOnly);
        (0..mask.len())
            .filter(|&i| mask[i] == 1.0)
            .find(|&i| matches!(
                decode_action_id(i),
                Some(CandidateAction::PlayCard { hand_index: hi, target_code: tc, .. })
                    if hi == hand_index && tc == target_code
            ))
            .unwrap_or_else(|| panic!(
                "no legal play action for hand_index={hand_index} target_code={target_code}; legal={:?}",
                legal_action_ids(&mask)
            ))
    }

    fn legal_action_ids(mask: &[f32]) -> Vec<usize> {
        (0..mask.len()).filter(|&i| mask[i] == 1.0).collect()
    }

    #[test]
    fn requires_target_recognizes_target_ally_max_hp_plus_variants() {
        // TAMHP-1: both universal and bare variants require a friendly target.
        assert!(requires_target(&["target_ally_max_hp_plus_universal_1".to_string()]));
        assert!(requires_target(&["target_ally_max_hp_plus_universal_7".to_string()]));
        assert!(requires_target(&["target_ally_max_hp_plus_3".to_string()]));
        // Mixed with irrelevant mechanics still requires.
        assert!(requires_target(&["shield".to_string(), "target_ally_max_hp_plus_universal_2".to_string()]));
        // Non-TAMHP mechanics alone do not require (sanity: no false positive).
        assert!(!requires_target(&["shield".to_string()]));
        assert!(!requires_target(&["aoe_silence".to_string()]));
        assert!(!requires_target(&["team_wide_shield".to_string()]));
    }

    #[test]
    fn apply_play_aoe_silence_strips_up_to_three_enemy_mechanics() {
        // AOE-SILENCE-1 (card 47 Солдатик): strip mechanics from up to 3 enemy
        // minions that have mechanics. 3 enemy Сукуна (cleave_1_2) → all cleared.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(47, 7, 4, 5, vec!["aoe_silence"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p2.board = vec![
            card_with_mechanics(23, 7, 7, 5, vec!["cleave_1_2"]),
            card_with_mechanics(23, 7, 7, 5, vec!["cleave_1_2"]),
            card_with_mechanics(23, 7, 7, 5, vec!["cleave_1_2"]),
        ];
        // aoe_silence does not require a target → target_code 0.
        let action_id = find_play_action(&state, 1, 0, 0);
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let new = kernel_apply(&state, 1, action_id, &mut draw_rng);
        // All 3 enemy minions stripped of mechanics (cleave_1_2 removed).
        assert_eq!(new.p2.board.len(), 3);
        for u in &new.p2.board {
            assert!(u.mechanics.is_empty(), "enemy minion should be silenced, got {:?}", u.mechanics);
        }
        // Status flags untouched (is_frozen false here anyway — sanity).
        // The played Солдатик retains its own aoe_silence mechanic (silence
        // does NOT self-strip — it targets enemy minions only).
        let soldatik = new.p1.board.iter().find(|c| c.card_id == 47).unwrap();
        assert!(soldatik.mechanics.iter().any(|m| m == "aoe_silence"));
    }

    #[test]
    fn apply_play_aoe_silence_respects_limit_and_skips_mechanicless_units() {
        // 4 enemy minions with mechanics + 1 without. limit=3 → only 3
        // stripped; the mechanicless unit is skipped (does not consume limit).
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(47, 7, 4, 5, vec!["aoe_silence"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p2.board = vec![
            card_with_mechanics(23, 7, 7, 5, vec!["cleave_1_2"]),
            card_with_mechanics(27, 1, 2, 1, vec![]),           // no mechanics — skipped
            card_with_mechanics(24, 9, 5, 5, vec!["shield"]),
            card_with_mechanics(23, 7, 7, 5, vec!["cleave_1_2"]),
            card_with_mechanics(23, 7, 7, 5, vec!["cleave_1_2"]),
        ];
        let action_id = find_play_action(&state, 1, 0, 0);
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let new = kernel_apply(&state, 1, action_id, &mut draw_rng);
        // Exactly 3 of the 4-with-mechanics stripped; one retains its mechanic
        // (the 4th in board order, since silence iterates front-to-back and
        // the mechanicless unit at index 1 is skipped without consuming limit).
        let silenced_count = new.p2.board.iter().filter(|u| u.mechanics.is_empty()).count();
        assert_eq!(silenced_count, 4, "3 stripped + 1 originally-mechanicless = 4 empty");
        let with_mechanics: Vec<&KernelCard> = new.p2.board.iter().filter(|u| !u.mechanics.is_empty()).collect();
        assert_eq!(with_mechanics.len(), 1, "exactly one enemy retains mechanics (limit=3)");
        assert_eq!(with_mechanics[0].card_id, 23, "the 4th cleave unit (index 4) is the one not silenced");
    }

    #[test]
    fn apply_play_team_wide_shield_grants_shield_to_up_to_three_allies_excluding_self() {
        // TWS-1/2 (card 48 Соул Гудман): shield up to 3 friendly minions,
        // EXCLUDING the just-played card. 3 Скелет on board → all 3 get shield;
        // Соул Гудман does NOT (self-exclusion — TWS-2).
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(48, 7, 2, 4, vec!["team_wide_shield"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p1.board = vec![
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
        ];
        let action_id = find_play_action(&state, 1, 0, 0);
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let new = kernel_apply(&state, 1, action_id, &mut draw_rng);
        // 3 Скелет + Соул Гудман on board (cap 5 ok).
        let skeletons: Vec<&KernelCard> = new.p1.board.iter().filter(|c| c.card_id == 27).collect();
        assert_eq!(skeletons.len(), 3);
        for s in &skeletons {
            assert!(s.has_mechanic("shield"), "Скелет should gain shield, got {:?}", s.mechanics);
        }
        let soul = new.p1.board.iter().find(|c| c.card_id == 48).unwrap();
        assert!(!soul.has_mechanic("shield"), "Соул Гудман should self-exclude (no shield), got {:?}", soul.mechanics);
    }

    #[test]
    fn apply_play_team_wide_shield_respects_limit_of_three() {
        // 4 friendly unshielded minions on board (board cap 5 → playing Соул
        // Гудман makes 5). limit=3 → exactly 3 gain shield, 1 remains
        // unshielded (beyond limit). Played card is self-excluded and NOT
        // counted. Mirrors core/effects.py `_grant_shields(limit=3)`.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(48, 7, 2, 4, vec!["team_wide_shield"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p1.board = vec![
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
        ];
        let action_id = find_play_action(&state, 1, 0, 0);
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let new = kernel_apply(&state, 1, action_id, &mut draw_rng);
        let skeletons: Vec<&KernelCard> = new.p1.board.iter().filter(|c| c.card_id == 27).collect();
        assert_eq!(skeletons.len(), 4);
        let shielded = skeletons.iter().filter(|c| c.has_mechanic("shield")).count();
        assert_eq!(shielded, 3, "exactly 3 of 4 gain shield (limit=3); 1 beyond limit");
        let soul = new.p1.board.iter().find(|c| c.card_id == 48).unwrap();
        assert!(!soul.has_mechanic("shield"), "Соул Гудман self-excluded");
    }

    #[test]
    fn apply_play_tamhp_universal_bumps_friendly_minion_max_hp_no_clamp() {
        // TAMHP-1/2/3 (card 52 Криста Ленц, target_ally_max_hp_plus_universal_1):
        // target a friendly minion (code 9) → max_hp += 1, hp UNCHANGED (direct
        // increase, no heal_card clamp). Verify hp stays below new max_hp.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(52, 2, 1, 2, vec!["target_ally_max_hp_plus_universal_1"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        // Friendly minion with hp=1, max_hp=3 (damaged). Bumping max_hp to 4
        // must NOT raise hp (no heal_card clamp) — hp stays 1.
        state.p1.board = vec![card_with_mechanics(27, 1, 2, 1, vec![])];
        state.p1.board[0].max_hp = 3;
        let action_id = find_play_action(&state, 1, 0, 9);
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let new = kernel_apply(&state, 1, action_id, &mut draw_rng);
        // Board: [Скелет (target), Криста (played)]. Скелет is index 0.
        let skel = new.p1.board.iter().find(|c| c.card_id == 27).unwrap();
        assert_eq!(skel.max_hp, 4, "max_hp 3 + 1 = 4 (direct increase)");
        assert_eq!(skel.hp, 1, "hp UNCHANGED (no heal_card clamp) — audit risk note honored");
    }

    #[test]
    fn apply_play_tamhp_universal_bumps_own_hero_max_hp() {
        // TAMHP universal allows own hero (code 16). hero max_hp 30 → 31, hp
        // unchanged.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(52, 2, 1, 2, vec!["target_ally_max_hp_plus_universal_1"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        // No friendly minions → only own-hero target (code 16) is legal.
        let action_id = find_play_action(&state, 1, 0, 16);
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let hero_before = state.p1.hero.max_hp;
        let hero_hp_before = state.p1.hero.hp;
        let new = kernel_apply(&state, 1, action_id, &mut draw_rng);
        assert_eq!(new.p1.hero.max_hp, hero_before + 1, "hero max_hp 30 + 1 = 31");
        assert_eq!(new.p1.hero.hp, hero_hp_before, "hero hp unchanged");
    }

    #[test]
    fn apply_play_tamhp_playable_both_sides() {
        // User decision: card52 playable on BOTH sides. p2 plays Криста
        // targeting its friendly minion (code 9 from p2's perspective) →
        // p2's Скелет max_hp bumps. Verifies the target plumbing works for
        // whichever player plays it (AC-FFI-4).
        let mut state = KernelState {
            current_turn_owner_id: 2,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), Vec::new()),
            p2: player_with(Vec::new(), vec![card_with_mechanics(52, 2, 1, 2, vec!["target_ally_max_hp_plus_universal_1"])]),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p2.board = vec![card_with_mechanics(27, 1, 2, 1, vec![])];
        state.p2.board[0].max_hp = 3;
        // p2 is the current turn owner; find a legal play for p2 targeting
        // friendly board[0] (code 9).
        let action_id = find_play_action(&state, 2, 0, 9);
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let new = kernel_apply(&state, 2, action_id, &mut draw_rng);
        let skel = new.p2.board.iter().find(|c| c.card_id == 27).unwrap();
        assert_eq!(skel.max_hp, 4, "p2's Скелет max_hp 3 + 1 = 4");
        assert_eq!(skel.hp, 1, "hp unchanged");
    }

    #[test]
    fn mask_targets_for_card_tamhp_universal_enables_friendly_minion_and_hero_codes() {
        // Frozen-classic additive guard: card52's mask enables friendly-minion
        // codes 9..15 (for existing minions) + own-hero code 16. Verifies the
        // additive branch in mask_targets_for_card and that card52 is playable
        // (at least the hero target) even with an empty board.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(52, 2, 1, 2, vec!["target_ally_max_hp_plus_universal_1"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p1.board = vec![
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
        ];
        let mask = build_action_mask(&state, 1, PlacementMode::AppendOnly);
        // The play action for hand_index=0 (Криста) at the AppendOnly position
        // with each valid target_code must be legal.
        for tc in [9_usize, 10, 16] {
            let found = (0..mask.len())
                .filter(|&i| mask[i] == 1.0)
                .any(|i| matches!(
                    decode_action_id(i),
                    Some(CandidateAction::PlayCard { hand_index: 0, target_code, .. }) if target_code == tc
                ));
            assert!(found, "target_code {tc} should be legal for card52");
        }
        // target_code 0 (no target) must NOT be legal — card52 requires a target.
        let no_tgt = (0..mask.len())
            .filter(|&i| mask[i] == 1.0)
            .any(|i| matches!(
                decode_action_id(i),
                Some(CandidateAction::PlayCard { hand_index: 0, target_code: 0, .. })
            ));
        assert!(!no_tgt, "card52 must require a target (target_code 0 illegal)");
    }

    /// Helper: apply an action via RolloutKernel and return the new state.
    fn kernel_apply(
        state: &KernelState,
        player_id: i32,
        action_id: usize,
        draw_rng: &mut DrawRng,
    ) -> KernelState {
        let kernel = RolloutKernel::new(KernelConfig::default());
        kernel
            .apply_action(state, player_id, action_id, false, draw_rng)
            .expect("action applies")
            .state
    }

    // ---- Phase 4: cleave / instant_kill / freeze / armor_X_Y ----

    /// Build a minimal 2-player state with `me_board`/`opp_board` and a ready
    /// attacker at me_board[0]. `me_hero_id`/`opp_hero_id` select the hero
    /// card_id (default 0 = dummy 30hp hero).
    fn attack_state(me_board: Vec<KernelCard>, opp_board: Vec<KernelCard>) -> KernelState {
        let mut p1 = player_with(Vec::new(), Vec::new());
        p1.user_id = 1;
        p1.board = me_board;
        for c in p1.board.iter_mut() {
            c.is_ready = true;
        }
        let mut p2 = player_with(Vec::new(), Vec::new());
        p2.user_id = 2;
        p2.board = opp_board;
        KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1,
            p2,
            ..Default::default()
        }
    }

    fn find_attack_action(state: &KernelState, attacker_idx: usize, target_code: usize) -> usize {
        let mask = build_action_mask(state, 1, PlacementMode::AppendOnly);
        (0..mask.len())
            .filter(|&i| mask[i] == 1.0)
            .find(|&i| matches!(
                decode_action_id(i),
                Some(CandidateAction::Attack { attacker_index, target_code: tc })
                    if attacker_index == attacker_idx && tc == target_code
            ))
            .expect("attack action is legal")
    }

    #[test]
    fn apply_attack_cleave_splashes_damage_to_neighbours() {
        // Сукуна (7/7, cleave_1) attacks the middle Скелет (2hp). The main
        // target dies (2-7→0); cleave splashes 1 to each living neighbour
        // (2hp → 1hp). Сукуна takes 3 counter → 4hp.
        let me = vec![card_with_mechanics(23, 7, 7, 7, vec!["cleave_1"])];
        let opp = vec![
            card_with_mechanics(27, 1, 3, 2, vec![]),
            card_with_mechanics(27, 1, 3, 2, vec![]),
            card_with_mechanics(27, 1, 3, 2, vec![]),
        ];
        let state = attack_state(me, opp);
        let action_id = find_attack_action(&state, 0, 1);
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng).expect("attack");
        let new = out.state;
        // cleanup removes the dead middle Скелет; two neighbours remain at 1hp.
        assert_eq!(new.p2.board.len(), 2, "middle target dead, neighbours survive");
        assert!(new.p2.board.iter().all(|c| c.hp == 1), "neighbours at 1hp after cleave");
        assert_eq!(new.p1.board[0].hp, 4, "Сукуна took 3 counter damage");
    }

    #[test]
    fn apply_attack_cleave_skips_dead_neighbours() {
        // A dead neighbour (hp 0, pre-cleanup) is skipped by cleave; the alive
        // neighbour on the other side still takes the splash. Board layout:
        // [dead(0hp), target(2hp), alive(2hp)]. Сукуна attacks index 1.
        // Cleave hits index 0 (dead → skipped) and index 2 (alive 2→1).
        let me = vec![card_with_mechanics(23, 7, 7, 7, vec!["cleave_1"])];
        let mut opp_dead = card_with_mechanics(27, 1, 3, 1, vec![]);
        opp_dead.hp = 0;
        let opp = vec![
            opp_dead,
            card_with_mechanics(27, 1, 3, 2, vec![]),
            card_with_mechanics(27, 1, 3, 2, vec![]),
        ];
        let state = attack_state(me, opp);
        let action_id = find_attack_action(&state, 0, 1);
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng).expect("attack");
        let new = out.state;
        // After cleanup: dead index 0 + dead target index 1 removed; the
        // alive neighbour (was index 2) survives at 1hp.
        let survivors: Vec<i32> = new.p2.board.iter().map(|c| c.hp).collect();
        assert_eq!(survivors, vec![1], "alive neighbour took cleave (2-1=1), dead neighbour skipped");
    }

    #[test]
    fn apply_attack_instant_kill_sets_hp_zero_and_one_shot_flag() {
        // Сайтама (10/10, instant_kill) attacks a 15hp minion. Normal damage
        // would leave 5hp; instant_kill sets hp=0. instant_kill_used becomes
        // true (one-shot).
        let me = vec![card_with_mechanics(25, 10, 10, 10, vec!["instant_kill"])];
        let opp = vec![card_with_mechanics(42, 6, 15, 15, vec![])];
        let state = attack_state(me, opp);
        let action_id = find_attack_action(&state, 0, 0);
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng).expect("attack");
        let new = out.state;
        assert_eq!(new.p2.board.len(), 0, "target died via instant_kill (hp=0)");
        assert_eq!(new.p2.graveyard.len(), 1, "target in graveyard");
        // Сайтама died to 15 counter-attack (10-15→0).
        assert_eq!(new.p1.board.len(), 0, "Сайтама died to counter");
    }

    #[test]
    fn instant_kill_is_one_shot_per_card() {
        // Two separate attacks by the same Сайтама: only the FIRST triggers
        // instant_kill. Verify via direct apply_attack calls (no cleanup
        // between, so we can inspect hp on the surviving second target).
        // Use a target with hp > 10 and atk 0 so Сайтама survives.
        let me = vec![card_with_mechanics(25, 10, 10, 10, vec!["instant_kill"])];
        // First target: hp 12, atk 0 (dies to instant_kill, no counter).
        // Second target: hp 12, atk 0 (should survive at 2hp — normal damage
        // only, instant_kill already used).
        let opp = vec![
            card_with_mechanics(42, 6, 0, 12, vec![]),
            card_with_mechanics(42, 6, 0, 12, vec![]),
        ];
        let state = attack_state(me, opp);
        // First attack: target index 0.
        let aid1 = find_attack_action(&state, 0, 0);
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        let out1 = kernel.apply_action(&state, 1, aid1, false, &mut draw_rng).expect("attack1");
        // After first attack: target 0 hp=0 (instant_kill), Сайтама hp=10
        // (no counter, atk 0). Cleanup removes target 0.
        assert_eq!(out1.state.p2.board.len(), 1, "first target killed");
        assert!(out1.state.p1.board[0].instant_kill_used, "instant_kill_used set after first kill");
        // Re-ready Сайтама for the second attack (apply_attack sets is_ready=false).
        let mut state2 = out1.state.clone();
        state2.p1.board[0].is_ready = true;
        // Second attack: the remaining target (now index 0).
        let aid2 = find_attack_action(&state2, 0, 0);
        let out2 = kernel.apply_action(&state2, 1, aid2, false, &mut draw_rng).expect("attack2");
        // Without instant_kill (one-shot used), normal damage: 12-10=2hp.
        assert_eq!(out2.state.p2.board.len(), 1, "second target survived (no instant_kill)");
        assert_eq!(out2.state.p2.board[0].hp, 2, "second target took normal damage only (12-10=2)");
    }

    #[test]
    fn apply_attack_instant_kill_does_not_work_on_hero() {
        // Сайтама attacks the hero — instant_kill does NOT trigger; hero takes
        // only base damage.
        let me = vec![card_with_mechanics(25, 10, 10, 10, vec!["instant_kill"])];
        let state = attack_state(me, Vec::new());
        // p2 hero default hp 30.
        let action_id = find_attack_action(&state, 0, 7);
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng).expect("hero attack");
        assert_eq!(out.state.p2.hero.hp, 20, "hero took 10 base damage (no instant_kill on heroes)");
        // instant_kill_used stays false (not consumed on hero attack).
        assert!(!out.state.p1.board[0].instant_kill_used, "instant_kill not consumed on hero attack");
    }

    #[test]
    fn apply_play_battlecry_freeze_sets_is_frozen_on_target() {
        // Саб-Зиро (battlecry_freeze) played targeting enemy board[0] → frozen.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(19, 4, 3, 4, vec!["battlecry_freeze"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p1.mana = 10;
        state.p2.board = vec![card_with_mechanics(27, 1, 2, 1, vec![])];
        // Find the play action targeting enemy board[0] (target_code=1).
        let mask = build_action_mask(&state, 1, PlacementMode::AppendOnly);
        let action_id = (0..mask.len())
            .filter(|&i| mask[i] == 1.0)
            .find(|&i| matches!(
                decode_action_id(i),
                Some(CandidateAction::PlayCard { hand_index: 0, target_code: 1, .. })
            ))
            .expect("freeze play action is legal");
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng).expect("play");
        assert!(out.state.p2.board[0].is_frozen, "target frozen by battlecry_freeze");
    }

    #[test]
    fn apply_play_freeze_blocked_by_shield() {
        // A shielded target: freeze is blocked, shield consumed, NOT frozen.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(19, 4, 3, 4, vec!["battlecry_freeze"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p1.mana = 10;
        state.p2.board = vec![card_with_mechanics(17, 3, 3, 2, vec!["shield"])];
        let mask = build_action_mask(&state, 1, PlacementMode::AppendOnly);
        let action_id = (0..mask.len())
            .filter(|&i| mask[i] == 1.0)
            .find(|&i| matches!(
                decode_action_id(i),
                Some(CandidateAction::PlayCard { hand_index: 0, target_code: 1, .. })
            ))
            .expect("freeze play action is legal");
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng).expect("play");
        assert!(!out.state.p2.board[0].is_frozen, "freeze blocked by shield");
        assert!(!out.state.p2.board[0].has_mechanic("shield"), "shield consumed");
    }

    #[test]
    fn apply_play_aoe_freeze_freezes_up_to_three_enemies() {
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(22, 8, 6, 6, vec!["aoe_freeze"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p1.mana = 10;
        state.p2.board = vec![
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
        ];
        let mask = build_action_mask(&state, 1, PlacementMode::AppendOnly);
        let action_id = (0..mask.len())
            .filter(|&i| mask[i] == 1.0)
            .find(|&i| matches!(
                decode_action_id(i),
                Some(CandidateAction::PlayCard { hand_index: 0, target_code: 0, .. })
            ))
            .expect("aoe_freeze play action is legal");
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng).expect("play");
        let frozen = out.state.p2.board.iter().filter(|c| c.is_frozen).count();
        assert_eq!(frozen, 3, "aoe_freeze freezes up to first 3 enemy minions");
    }

    #[test]
    fn apply_end_turn_thaws_frozen_unit_and_skips_activation() {
        // A frozen unit on the opponent's board: when end_turn runs for the
        // current player, the opponent's frozen unit thaws (is_frozen=false)
        // and stays not-ready (skips one activation).
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), Vec::new()),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        let mut frozen_skel = card_with_mechanics(27, 1, 2, 1, vec![]);
        frozen_skel.is_frozen = true;
        frozen_skel.is_ready = true; // was ready before freeze
        state.p2.board = vec![frozen_skel];
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut rng = ChaChaRng::seed_from_u64(42);
        let mut draw_rng = DrawRng::live(&mut rng);
        // p1 ends turn → p2's turn begins; p2's frozen unit thaws + skips.
        let out = kernel.apply_action(&state, 1, 0, false, &mut draw_rng).expect("end turn");
        assert!(!out.state.p2.board[0].is_frozen, "frozen unit thawed at end-turn");
        assert!(!out.state.p2.board[0].is_ready, "thawed unit skips activation (is_ready=false)");
    }

    #[test]
    fn armor_x_y_rolls_randint_via_recorded_rng() {
        // Direct test of the armor_X_Y range roll: a target with armor_1_3
        // takes damage reduced by a recorded randint roll. Uses a Recorded
        // DrawRng with randint_rolls=[2] → armor absorbs 2.
        let mut target = card_with_mechanics(5, 0, 0, 10, vec!["armor_1_3"]);
        target.mechanics = vec!["armor_1_3".to_string()];
        let mut draw_rng = DrawRng::recorded(vec![], vec![], vec![2], vec![]);
        let dmg = apply_damage_modifiers(&mut target, 7, &mut draw_rng);
        // 7 - 2 (rolled armor) = 5.
        assert_eq!(dmg, 5, "armor_1_3 rolled 2 → damage 7-2=5");
    }

    #[test]
    fn armor_x_fixed_does_not_roll() {
        // armor_X (single value) is a fixed reduction, no randint roll.
        let mut target = card_with_mechanics(18, 5, 5, 5, vec!["armor_1"]);
        target.mechanics = vec!["armor_1".to_string()];
        // Recorded with NO randint_rolls — if armor_1 incorrectly rolled, the
        // stream would be exhausted (fall back to min=1) but still produce 1.
        // The key check: damage = 7 - 1 = 6 (fixed), no roll consumed.
        let mut draw_rng = DrawRng::recorded(vec![], vec![], vec![], vec![]);
        let dmg = apply_damage_modifiers(&mut target, 7, &mut draw_rng);
        assert_eq!(dmg, 6, "armor_1 fixed → damage 7-1=6");
    }

    #[test]
    fn armor_x_y_recorded_stream_exhaustion_falls_back_to_min() {
        // If the recorded randint stream is exhausted, roll_range falls back
        // to min (matching the pick/order exhaustion contract).
        let mut target = card_with_mechanics(5, 0, 0, 10, vec!["armor_1_3"]);
        target.mechanics = vec!["armor_1_3".to_string()];
        let mut draw_rng = DrawRng::recorded(vec![], vec![], vec![], vec![]);
        let dmg = apply_damage_modifiers(&mut target, 7, &mut draw_rng);
        // Fall back to min=1 → 7-1=6.
        assert_eq!(dmg, 6, "exhausted randint stream falls back to min=1");
    }

    #[test]
    fn roll_range_live_uses_gen_range() {
        // Live DrawRng produces a value within [min, max].
        let mut rng = ChaChaRng::seed_from_u64(7);
        let mut draw_rng = DrawRng::live(&mut rng);
        for _ in 0..20 {
            let v = roll_range(1, 3, &mut draw_rng);
            assert!((1..=3).contains(&v), "live roll_range in [1,3]: {}", v);
        }
    }

    #[test]
    fn roll_range_recorded_pops_in_order() {
        let mut draw_rng = DrawRng::recorded(vec![], vec![], vec![3, 1, 2], vec![]);
        assert_eq!(roll_range(1, 3, &mut draw_rng), 3);
        assert_eq!(roll_range(1, 3, &mut draw_rng), 1);
        assert_eq!(roll_range(1, 3, &mut draw_rng), 2);
    }

    // ---- Phase 6: random_battlecry (card15) + shield_refresh (card24) ----

    #[test]
    fn is_random_battlecry_damage_card_is_mechanic_driven_not_card_id() {
        // Any card carrying `battlecry_damage_X_random` is a random-battlecry-
        // damage card — mechanic-driven, NOT card_id == 15.
        let card15 = card_with_mechanics(15, 2, 2, 1, vec!["battlecry_damage_1_random"]);
        assert!(is_random_battlecry_damage_card(&card15), "card15 matches");
        // A different card_id with the same mechanic also matches (future-proof).
        let other = card_with_mechanics(999, 2, 2, 1, vec!["battlecry_damage_3_random"]);
        assert!(is_random_battlecry_damage_card(&other), "card999 with mechanic matches");
        // A targeted battlecry_damage card does NOT match.
        let targeted = card_with_mechanics(19, 2, 2, 1, vec!["battlecry_damage_1"]);
        assert!(!is_random_battlecry_damage_card(&targeted), "targeted does not match");
        // No-mechanic card does NOT match.
        let plain = card_with_mechanics(27, 1, 2, 1, vec![]);
        assert!(!is_random_battlecry_damage_card(&plain), "plain does not match");
        // card_id == 15 but WITHOUT the mechanic does NOT match (mechanic is
        // the sole discriminator, not the id).
        let id_only = card_with_mechanics(15, 2, 2, 1, vec![]);
        assert!(!is_random_battlecry_damage_card(&id_only), "card_id 15 without mechanic does not match");
    }

    #[test]
    fn roll_choice_recorded_pops_index_in_order() {
        // Recorded choice_rolls pop in call order; clamp keeps stale entries
        // in-bounds.
        let mut draw_rng = DrawRng::recorded(vec![], vec![], vec![], vec![2, 0, 1]);
        assert_eq!(roll_choice(3, &mut draw_rng), 2);
        assert_eq!(roll_choice(3, &mut draw_rng), 0);
        assert_eq!(roll_choice(3, &mut draw_rng), 1);
    }

    #[test]
    fn roll_choice_recorded_clamps_out_of_range_index() {
        // A stale fixture entry larger than n-1 is clamped to n-1 (robust
        // replay, mirrors the pick/order exhaustion contract).
        let mut draw_rng = DrawRng::recorded(vec![], vec![], vec![], vec![99]);
        assert_eq!(roll_choice(3, &mut draw_rng), 2, "99 clamped to 2 (n-1)");
    }

    #[test]
    fn roll_choice_recorded_exhaustion_falls_back_to_zero() {
        // Empty stream → fall back to 0.
        let mut draw_rng = DrawRng::recorded(vec![], vec![], vec![], vec![]);
        assert_eq!(roll_choice(3, &mut draw_rng), 0, "exhausted stream falls back to 0");
    }

    #[test]
    fn roll_choice_live_uses_gen_range() {
        let mut rng = ChaChaRng::seed_from_u64(11);
        let mut draw_rng = DrawRng::live(&mut rng);
        for _ in 0..20 {
            let v = roll_choice(5, &mut draw_rng);
            assert!((0..5).contains(&v), "live roll_choice in [0,5): {}", v);
        }
    }

    #[test]
    fn random_battlecry_damage_uses_recorded_choice_roll() {
        // card15-equivalent: a card with `battlecry_damage_1_random` is played
        // against an opponent with 2 minions (board) + hero → targets list has
        // 3 entries. Recorded choice_rolls=[1] → the 2nd minion (board[1]) is
        // hit for 1 damage. Mechanic-driven (card_id 999, not 15).
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(999, 2, 2, 1, vec!["battlecry_damage_1_random"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        // opponent (p2) board: 2 minions with 2 hp each (so 1 damage doesn't
        // kill — easy to assert the hit).
        state.p2.board = vec![
            card_with_mechanics(27, 1, 2, 2, vec![]),
            card_with_mechanics(27, 1, 2, 2, vec![]),
        ];
        let kernel = RolloutKernel::new(KernelConfig::default());
        // Play the random-battlecry card (hand_index 0, no target required —
        // random battlecry does not need a target). Find the legal play action
        // with target_code 0.
        let action_id = find_play_action(&state, 1, 0, 0);
        let mut draw_rng = DrawRng::recorded(vec![], vec![], vec![], vec![1]);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng);
        assert!(out.is_ok(), "random battlecry play applies: {:?}", out.err());
        let new = out.unwrap().state;
        // targets = list(opponent.board) + [opponent.hero]; index 1 = board[1].
        assert_eq!(new.p2.board[0].hp, 2, "board[0] untouched (choice index 1)");
        assert_eq!(new.p2.board[1].hp, 1, "board[1] took 1 damage (choice index 1)");
        assert_eq!(new.p2.hero.hp, 30, "hero untouched (choice index 1)");
        // The played card is on p1's board.
        assert_eq!(new.p1.board.len(), 1);
        assert_eq!(new.p1.board[0].card_id, 999);
    }

    #[test]
    fn random_battlecry_damage_can_target_hero() {
        // Same mechanic, recorded choice_rolls=[2] → index 2 = opponent hero.
        // opponent board has 2 minions (indices 0,1) + hero (index 2).
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(999, 2, 2, 1, vec!["battlecry_damage_1_random"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p2.board = vec![
            card_with_mechanics(27, 1, 2, 2, vec![]),
            card_with_mechanics(27, 1, 2, 2, vec![]),
        ];
        let kernel = RolloutKernel::new(KernelConfig::default());
        let action_id = find_play_action(&state, 1, 0, 0);
        let mut draw_rng = DrawRng::recorded(vec![], vec![], vec![], vec![2]);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng);
        assert!(out.is_ok(), "random battlecry play applies: {:?}", out.err());
        let new = out.unwrap().state;
        assert_eq!(new.p2.board[0].hp, 2, "board[0] untouched (choice index 2 = hero)");
        assert_eq!(new.p2.board[1].hp, 2, "board[1] untouched (choice index 2 = hero)");
        assert_eq!(new.p2.hero.hp, 29, "hero took 1 damage (choice index 2)");
    }

    #[test]
    fn shield_refresh_mechanic_driven_end_of_turn_restores_shield() {
        // card24-equivalent: a unit with `shield_refresh` (and NO current
        // `shield`) on the opponent's board regains `shield` at the start of
        // its owner's turn (i.e. when the OTHER player ends turn). Mechanic-
        // driven (card_id 999, not 24).
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), Vec::new()),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        // p2's unit has shield_refresh but shield was already consumed (no
        // `shield` mechanic present).
        state.p2.board = vec![card_with_mechanics(999, 9, 5, 6, vec!["shield_refresh"])];
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut det_rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut det_rng);
        // p1 ends turn → p2's turn begins → p2's unit regains shield.
        let out = kernel.apply_action(&state, 1, 0, false, &mut draw_rng).expect("end turn");
        assert!(
            out.state.p2.board[0].mechanics.iter().any(|m| m == "shield"),
            "shield_refresh re-adds shield at owner turn start"
        );
        // shield_refresh itself is NOT consumed (persists, matches Python).
        assert!(
            out.state.p2.board[0].mechanics.iter().any(|m| m == "shield_refresh"),
            "shield_refresh mechanic persists"
        );
    }

    #[test]
    fn shield_refresh_does_not_double_add_when_shield_present() {
        // If the unit already has `shield`, shield_refresh does NOT add a
        // second shield (matches Python: `if "shield_refresh" in mechanics and
        // "shield" not in mechanics`).
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), Vec::new()),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p2.board = vec![card_with_mechanics(24, 9, 5, 6, vec!["shield", "shield_refresh"])];
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut det_rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut det_rng);
        let out = kernel.apply_action(&state, 1, 0, false, &mut draw_rng).expect("end turn");
        let shield_count = out.state.p2.board[0].mechanics.iter().filter(|m| *m == "shield").count();
        assert_eq!(shield_count, 1, "no duplicate shield added when already present");
    }

    #[test]
    fn shield_refresh_no_op_for_unit_without_mechanic() {
        // A unit WITHOUT shield_refresh does NOT gain shield at turn start.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), Vec::new()),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p2.board = vec![card_with_mechanics(27, 1, 2, 1, vec![])];
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut det_rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut det_rng);
        let out = kernel.apply_action(&state, 1, 0, false, &mut draw_rng).expect("end turn");
        assert!(
            !out.state.p2.board[0].mechanics.iter().any(|m| m == "shield"),
            "plain unit does not gain shield"
        );
    }

    // ---- Phase 7: sudden-death (WD-1) + max_turns truncation (WD-2) ----

    fn sudden_death_config() -> KernelConfig {
        let mut cfg = KernelConfig::default();
        cfg.sudden_death_enabled = true;
        cfg.sudden_death_damage_start = 1;
        cfg.sudden_death_damage_step = 1;
        cfg
    }

    #[test]
    fn sudden_death_applies_escalating_damage_to_active_player_at_end_turn() {
        // damage = damage_start + (turn_count-1)*damage_step, per-player
        // escalating, applied to the NEW active player at the start of their
        // turn (right after turn_number increment + owner switch). Direct hp
        // subtraction (no apply_damage). With start=1, step=1:
        //   end_turn#1 (p1 ends): turn 2, p2 tick#1 → 1 dmg.  p2: 30→29.
        //   end_turn#2 (p2 ends): turn 3, p1 tick#1 → 1 dmg.  p1: 30→29.
        //   end_turn#3 (p1 ends): turn 4, p2 tick#2 → 2 dmg.  p2: 29→27.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), Vec::new()),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        // Give each player a deck so the end-turn draw does not error out.
        state.p1.deck = vec![card(11, 1, 1, 1)];
        state.p2.deck = vec![card(12, 1, 1, 1)];
        let kernel = RolloutKernel::new(sudden_death_config());
        let mut det_rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut det_rng);

        // end_turn#1: p1 ends → p2's turn.
        let out = kernel
            .apply_action(&state, 1, 0, false, &mut draw_rng)
            .expect("end turn 1");
        assert_eq!(out.state.turn_number, 2);
        assert_eq!(out.state.current_turn_owner_id, 2);
        assert_eq!(out.state.p1.hero.hp, 30, "p1 untouched on p2's turn");
        assert_eq!(out.state.p2.hero.hp, 29, "p2 takes 1 (first tick)");
        assert_eq!(out.state.sudden_death_turns_by_player.get(&2), Some(&1));
        assert_eq!(out.state.sudden_death_last_applied_turn_by_player.get(&2), Some(&2));

        // end_turn#2: p2 ends → p1's turn.
        let out2 = kernel
            .apply_action(&out.state, 2, 0, false, &mut draw_rng)
            .expect("end turn 2");
        assert_eq!(out2.state.turn_number, 3);
        assert_eq!(out2.state.p1.hero.hp, 29, "p1 takes 1 (first tick)");
        assert_eq!(out2.state.p2.hero.hp, 29, "p2 unchanged on p1's turn");
        assert_eq!(out2.state.sudden_death_turns_by_player.get(&1), Some(&1));

        // end_turn#3: p1 ends → p2's turn. p2's SECOND tick → 2 damage.
        let out3 = kernel
            .apply_action(&out2.state, 1, 0, false, &mut draw_rng)
            .expect("end turn 3");
        assert_eq!(out3.state.turn_number, 4);
        assert_eq!(out3.state.p2.hero.hp, 27, "p2 takes 2 (second tick, escalation)");
        assert_eq!(out3.state.sudden_death_turns_by_player.get(&2), Some(&2));
        assert_eq!(out3.state.sudden_death_last_applied_turn_by_player.get(&2), Some(&4));
    }

    #[test]
    fn sudden_death_kills_hero_and_skips_mana_board_draw() {
        // When sudden-death damage drops the hero to <=0, check_game_over
        // flips status and apply_end_turn returns early WITHOUT restoring
        // mana, readying the board, or drawing (mirrors Python's
        // `if self.state.status != GameStatus.ONGOING: return`).
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), Vec::new()),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        // p2 hero at 1 HP — a single tick (1 dmg) kills it. p2 mana 0/0 so a
        // normal end_turn would restore mana to 1; on death it must stay 0.
        state.p2.hero.hp = 1;
        state.p2.mana = 0;
        state.p2.max_mana = 0;
        state.p2.deck = vec![card(12, 1, 1, 1)];
        // p2 has a frozen unit that would be thawed+readied on a normal turn
        // start — on death it must remain frozen/asleep.
        let mut frozen_unit = card_with_mechanics(27, 1, 1, 1, vec![]);
        frozen_unit.is_frozen = true;
        frozen_unit.is_ready = false;
        state.p2.board = vec![frozen_unit];

        let kernel = RolloutKernel::new(sudden_death_config());
        let mut det_rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut det_rng);
        let out = kernel
            .apply_action(&state, 1, 0, false, &mut draw_rng)
            .expect("end turn");

        assert_eq!(out.state.status, "p1_win", "p2 hero died → p1 wins");
        assert_eq!(out.state.p2.hero.hp, 0, "p2 hero dropped to 0");
        assert!(out.terminated, "terminated flag set on death");
        assert!(!out.truncated, "no truncation on death");
        // Early-return guards: mana NOT restored, board NOT readied, NO draw.
        assert_eq!(out.state.p2.mana, 0, "mana not restored after sudden-death death");
        assert_eq!(out.state.p2.max_mana, 0, "max_mana not incremented after death");
        assert!(out.state.p2.board[0].is_frozen, "frozen unit not thawed after death");
        assert!(!out.state.p2.board[0].is_ready, "unit not readied after death");
        assert_eq!(out.state.p2.deck.len(), 1, "no draw after sudden-death death");
    }

    #[test]
    fn sudden_death_disabled_by_default_is_no_op() {
        // Default KernelConfig has sudden_death_enabled=false → end_turn must
        // NOT touch hero hp or the sudden-death counters (matches Python
        // ClassicRLEnv which never passes classic_params).
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), Vec::new()),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p2.deck = vec![card(12, 1, 1, 1)];
        let kernel = RolloutKernel::new(KernelConfig::default());
        let mut det_rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut det_rng);
        let out = kernel
            .apply_action(&state, 1, 0, false, &mut draw_rng)
            .expect("end turn");
        assert_eq!(out.state.p1.hero.hp, 30);
        assert_eq!(out.state.p2.hero.hp, 30);
        assert!(out.state.sudden_death_turns_by_player.is_empty());
        assert!(out.state.sudden_death_last_applied_turn_by_player.is_empty());
    }

    #[test]
    fn max_turns_truncation_flags_truncated_when_turn_exceeds_limit() {
        // truncated = turn_number > max_turns (strictly greater-than, matching
        // Python `st.turn_number > self._max_turns`). Independent of
        // terminated (status != ongoing). With max_turns=4 and turn_number=4
        // pre-step, an end_turn → turn 5 > 4 → truncated=true, terminated=false.
        let mut cfg = KernelConfig::default();
        cfg.max_turns = 4;
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 4,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), Vec::new()),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p2.deck = vec![card(12, 1, 1, 1)];
        let kernel = RolloutKernel::new(cfg);
        let mut det_rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut det_rng);
        let out = kernel
            .apply_action(&state, 1, 0, false, &mut draw_rng)
            .expect("end turn");
        assert_eq!(out.state.turn_number, 5);
        assert!(!out.terminated, "game still ongoing → not terminated");
        assert!(out.truncated, "turn 5 > max_turns 4 → truncated");
    }

    #[test]
    fn max_turns_truncation_boundary_not_triggered_at_exact_limit() {
        // At turn_number == max_turns (not yet exceeded), an end_turn to
        // turn_number == max_turns must NOT truncate. With max_turns=4 and
        // pre-turn 3, end_turn → turn 4 == max_turns → truncated=false.
        let mut cfg = KernelConfig::default();
        cfg.max_turns = 4;
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 3,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), Vec::new()),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p2.deck = vec![card(12, 1, 1, 1)];
        let kernel = RolloutKernel::new(cfg);
        let mut det_rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut det_rng);
        let out = kernel
            .apply_action(&state, 1, 0, false, &mut draw_rng)
            .expect("end turn");
        assert_eq!(out.state.turn_number, 4);
        assert!(!out.truncated, "turn 4 == max_turns 4 → not truncated (strict >)");
    }

    // ---- Phase 7 fixes (FIX 2 / FIX 4 / FIX 7) ----

    #[test]
    fn mask_masks_out_consume_ally_play_when_board_full() {
        // CLN-3 full-board mask parity: board.len()==5 + a consume_ally
        // warrior in hand → the frozen codec mask
        // (classic_actions_v1._mask_play_actions:228) does NOT exempt
        // consume_ally, so the consume play action (targeting friendly
        // board[0], target_code=9) is masked OUT (bit == 0.0). Rust's
        // action_mask must match the frozen codec byte-for-byte. The APPLY
        // path (apply_play_card) and core/engine.py:1228 still EXEMPT
        // consume_ally (the play is accepted if forced via apply_action),
        // but the model mask never exposes it at a full board.
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(Vec::new(), vec![card_with_mechanics(20, 3, 2, 2, vec!["consume_ally"])]),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p1.mana = 10;
        state.p1.max_mana = 10;
        // Fill the board to the GAME_BOARD_CAP (5).
        state.p1.board = vec![
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
            card_with_mechanics(27, 1, 2, 1, vec![]),
        ];
        assert_eq!(state.p1.board.len(), GAME_BOARD_CAP);
        let mask = build_action_mask(&state, 1, PlacementMode::AppendOnly);
        // The consume_ally play action (hand=0, target_code=9) must be
        // masked OUT — the frozen codec masks consume_ally at a full board.
        let consume_action_id = (0..mask.len())
            .filter(|&i| mask[i] == 1.0)
            .find(|&i| matches!(
                decode_action_id(i),
                Some(CandidateAction::PlayCard { hand_index: 0, target_code: 9, .. })
            ));
        assert!(
            consume_action_id.is_none(),
            "consume_ally play action must be masked OUT at full board (frozen codec parity); legal={:?}",
            legal_action_ids(&mask)
        );
        // Sanity: NO play action for hand=0 is exposed at all (the board-full
        // guard skips the consume_ally card entirely in the mask loop).
        let any_hand0_play = (0..mask.len())
            .filter(|&i| mask[i] == 1.0)
            .any(|i| matches!(
                decode_action_id(i),
                Some(CandidateAction::PlayCard { hand_index: 0, .. })
            ));
        assert!(
            !any_hand0_play,
            "no play action for the consume_ally card should be exposed at full board; legal={:?}",
            legal_action_ids(&mask)
        );
    }

    #[test]
    fn roll_range_degenerate_range_consumes_recorded_stream() {
        // FIX 4: Python `random.randint(a, a)` (min>=max degenerate range,
        // e.g. armor_X_X) still consumes the RNG stream. Rust's `roll_range`
        // must pop the recorded value BEFORE the min>=max early-return so the
        // recorded stream stays aligned across degenerate rolls.
        let mut draw_rng = DrawRng::recorded(vec![], vec![], vec![42, 99], vec![]);
        // Degenerate range 5..=5: Python randint(5,5)==5 but still consumes.
        let v = roll_range(5, 5, &mut draw_rng);
        assert_eq!(v, 5, "degenerate range returns min");
        // The stream must have been consumed (one entry popped).
        match &draw_rng {
            DrawRng::Recorded { randint_rolls, .. } => {
                assert_eq!(
                    randint_rolls.len(),
                    1,
                    "degenerate roll_range must pop exactly one recorded entry (FIX 4)"
                );
                assert_eq!(randint_rolls.front(), Some(&99));
            }
            _ => panic!("expected Recorded DrawRng"),
        }
        // A subsequent non-degenerate roll consumes the next entry.
        let v2 = roll_range(0, 100, &mut draw_rng);
        assert_eq!(v2, 99, "next recorded entry consumed");
    }

    #[test]
    fn tamhp_full_mode_buffs_pre_insertion_target_not_played_card() {
        // FIX 7 (TAMHP Full-mode post-insertion indexing): when a TAMHP
        // warrior is played in Full placement mode at position <= the
        // target's pre-insertion index, the post-insertion index shifts
        // right by 1. The buff must land on the INTENDED target (the
        // pre-insertion friendly minion), NOT the just-played card. Python
        // resolves the target to a stable instance_id at decode time
        // (classic_actions_v1.py:84,122-123) and searches by instance_id
        // (effects.py:1126-1133); Rust has no instance_id, so it computes
        // the post-insertion index from the insert position.
        //
        // Setup: board = [A(27, 2/1), B(27, 2/1)], hand = [card52 Криста
        // (1/2, target_ally_max_hp_plus_universal_1)], mana=10. Play card52
        // targeting A (target_code=9 → pre_index 0) at position 0 (Full
        // mode). After insert: board = [card52, A, B]. A is now at index 1
        // (shifted). The buff (+1 max_hp) must hit A (board[1], max_hp 1→2),
        // NOT card52 (board[0], max_hp stays 2).
        let mut state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1: player_with(
                Vec::new(),
                vec![card_with_mechanics(52, 2, 1, 2, vec!["target_ally_max_hp_plus_universal_1"])],
            ),
            p2: player_with(Vec::new(), Vec::new()),
            ..Default::default()
        };
        state.p1.user_id = 1;
        state.p2.user_id = 2;
        state.p1.mana = 10;
        state.p1.max_mana = 10;
        state.p1.board = vec![
            card_with_mechanics(27, 1, 2, 1, vec![]), // A (target)
            card_with_mechanics(27, 1, 2, 1, vec![]), // B
        ];
        // Full placement mode: position 0 is legal. apply_action re-checks
        // the mask with the kernel's configured placement_mode, so the
        // kernel MUST be configured for Full mode (default is AppendOnly,
        // which only allows position == board.len()).
        let mut config = KernelConfig::default();
        config.placement_mode = PlacementMode::Full;
        let kernel = RolloutKernel::new(config);
        // Full placement mode: position 0 is legal. Find the PlayCard action
        // with hand_index=0, position=0, target_code=9 (friendly board[0]=A).
        let mask = build_action_mask(&state, 1, PlacementMode::Full);
        let action_id = (0..mask.len())
            .filter(|&i| mask[i] == 1.0)
            .find(|&i| matches!(
                decode_action_id(i),
                Some(CandidateAction::PlayCard { hand_index: 0, board_position: 0, target_code: 9 })
            ))
            .expect("TAMHP play@pos0 target=A is legal in Full mode");
        let mut rng = crate::worker::WorkerRng::Deterministic;
        let mut draw_rng = DrawRng::live(&mut rng);
        let out = kernel.apply_action(&state, 1, action_id, false, &mut draw_rng);
        assert!(out.is_ok(), "TAMHP play applies: {:?}", out.err());
        let new = out.unwrap().state;
        assert_eq!(new.p1.board.len(), 3, "played card inserted");
        // board[0] = the just-played card52 (max_hp UNCHANGED at 2).
        assert_eq!(new.p1.board[0].card_id, 52, "board[0] is the played card52");
        assert_eq!(new.p1.board[0].max_hp, 2, "played card52 max_hp NOT buffed");
        // board[1] = A (the intended target, shifted from pre_index 0).
        assert_eq!(new.p1.board[1].card_id, 27, "board[1] is A (target, shifted)");
        assert_eq!(new.p1.board[1].max_hp, 2, "A max_hp buffed 1→2 (FIX 7: post-insertion index)");
        // board[2] = B (unchanged).
        assert_eq!(new.p1.board[2].card_id, 27);
        assert_eq!(new.p1.board[2].max_hp, 1, "B unchanged");
    }
}
