use crate::env::TrainEnvConfig;
use crate::exploit::{select_exploit_action_for_state, ExploitAgentKind};
use crate::kernel::{
    action_type_for_id, build_history_event_v5, DrawRng, GoldenSnapshot, GoldenTraceConfig,
    KernelConfig, KernelHistoryEvent, KernelState, RolloutKernel,
};
use crate::v5::OBS_DIM_V5;
use crate::{ACTION_FEATURE_DIM_V1, MAX_CANDIDATE_ACTIONS, OBS_DIM_V1};

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaChaRng;
use rayon::prelude::*;

/// Per-slot RNG owned by the worker. The default is `Deterministic` (used
/// by golden-fixture replay and the bench bin) so the frozen fixtures —
/// whose draws are all `deck[0]` — reproduce exactly: a zero-RNG makes
/// `weighted_choice_idx` pick index 0 and `shuffle` a no-op for
/// single-card graveyards. Training calls `BatchedRolloutWorker::
/// use_chacha_rng` once at init to switch to `ChaCha` seeded from
/// `KernelConfig.seed` for real No-FIFO weighted draws.
#[derive(Debug, Clone)]
pub enum WorkerRng {
    ChaCha(ChaChaRng),
    Deterministic,
}

impl WorkerRng {
    fn cha_cha(seed: u64) -> Self {
        WorkerRng::ChaCha(ChaChaRng::seed_from_u64(seed))
    }
}

impl RngCore for WorkerRng {
    fn next_u32(&mut self) -> u32 {
        match self {
            WorkerRng::ChaCha(r) => r.next_u32(),
            WorkerRng::Deterministic => 0,
        }
    }
    fn next_u64(&mut self) -> u64 {
        match self {
            WorkerRng::ChaCha(r) => r.next_u64(),
            WorkerRng::Deterministic => 0,
        }
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        match self {
            WorkerRng::ChaCha(r) => r.fill_bytes(dest),
            WorkerRng::Deterministic => {
                for b in dest.iter_mut() {
                    *b = 0;
                }
            }
        }
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        match self {
            WorkerRng::ChaCha(r) => r.try_fill_bytes(dest),
            WorkerRng::Deterministic => {
                for b in dest.iter_mut() {
                    *b = 0;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub env_count: usize,
    pub max_steps_per_episode: usize,
    pub action_features_float16: bool,
    pub env: TrainEnvConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionFeatureOutput {
    DenseAndLegal,
    LegalOnly,
}

impl ActionFeatureOutput {
    fn include_dense(self) -> bool {
        matches!(self, Self::DenseAndLegal)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationOutput {
    V1AndV5,
    V5Only,
}

impl ObservationOutput {
    fn include_v1(self) -> bool {
        matches!(self, Self::V1AndV5)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionMaskOutput {
    Dense,
    LegalOnly,
}

impl ActionMaskOutput {
    fn include_dense(self) -> bool {
        matches!(self, Self::Dense)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalObservationOutput {
    Full,
    None,
}

impl TerminalObservationOutput {
    fn include_terminal_observations(self) -> bool {
        matches!(self, Self::Full)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticOutput {
    Full,
    None,
}

impl DiagnosticOutput {
    fn include_diagnostics(self) -> bool {
        matches!(self, Self::Full)
    }
}

fn validate_shared_config(expected: KernelConfig, candidate: KernelConfig) -> Result<(), String> {
    if candidate.placement_mode != expected.placement_mode {
        return Err("trace pool has incompatible placement_mode".to_string());
    }
    if candidate.mana_per_turn != expected.mana_per_turn {
        return Err("trace pool has incompatible mana_per_turn".to_string());
    }
    if candidate.v5_weighted_reward != expected.v5_weighted_reward {
        return Err("trace pool has incompatible v5_weighted_reward".to_string());
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct BatchTensorOutput {
    pub env_count: usize,
    pub observation_v1: Vec<f32>,
    pub observation_v5: Vec<f32>,
    pub action_mask: Vec<f32>,
    pub action_features: Vec<f32>,
    pub legal_action_counts: Vec<usize>,
    pub legal_action_offsets: Vec<usize>,
    pub legal_action_ids: Vec<usize>,
    pub legal_action_features: Vec<f32>,
    pub selected_local_indices: Vec<i32>,
    pub rewards: Vec<f32>,
    /// Exact reward for the non-acting player on each transition.
    pub counterparty_rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    /// Per-env truncation flag (WD-2): true when the post-step
    /// `turn_number > max_turns`, mirroring
    /// `ai/train_v2/classic_rl_env.py::ClassicRLEnv.step`'s `truncated`.
    /// Independent of `terminated` (which is `status != "ongoing"`),
    /// matching Python. Exposed via FFI for GAE bootstrap decisions.
    pub truncated: Vec<bool>,
    pub reset_flags: Vec<bool>,
    pub terminal_observation_v1: Vec<f32>,
    pub terminal_observation_v5: Vec<f32>,
    pub terminal_observation_valid: Vec<bool>,
    pub episode_returns: Vec<f32>,
    pub episode_lengths: Vec<usize>,
    /// Per-env parallel mana_draw head flag for the post-step observation
    /// (Phase 2: MD-3). True when mana_draw is a legal action this turn for
    /// the env's current actor. Exposed via FFI for the model/policy.
    pub mana_draw_legal: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct BatchedRolloutWorker {
    initial_states: Vec<KernelState>,
    initial_histories: Vec<Vec<KernelHistoryEvent>>,
    initial_configs: Vec<KernelConfig>,
    reset_pool_states: Option<Vec<KernelState>>,
    reset_pool_histories: Option<Vec<Vec<KernelHistoryEvent>>>,
    reset_pool_configs: Option<Vec<KernelConfig>>,
    next_reset_indices: Vec<usize>,
    states: Vec<KernelState>,
    histories: Vec<Vec<KernelHistoryEvent>>,
    slot_configs: Vec<KernelConfig>,
    rngs: Vec<WorkerRng>,
    episode_returns: Vec<f32>,
    episode_lengths: Vec<usize>,
    action_feature_output: ActionFeatureOutput,
    observation_output: ObservationOutput,
    action_mask_output: ActionMaskOutput,
    terminal_observation_output: TerminalObservationOutput,
    diagnostic_output: DiagnosticOutput,
}

#[derive(Debug, Clone)]
pub struct RuleAdvanceOutput {
    pub learner_rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    /// Per-env truncation flag (WD-2): true when the opponent's auto-advance
    /// step crossed `turn_number > max_turns`, mirroring
    /// `ai/train_v2/rollout_worker.py::_auto_play_until_learner`'s
    /// accumulated `truncated` (rollout_worker.py:~255). Independent of
    /// `terminated`; the caller resets on `terminated OR truncated`
    /// (rollout_worker.py:~375) BEFORE the learner acts. Populated from
    /// `step.truncated` in the advance loop, mirroring how `terminated` is
    /// collected.
    pub truncated: Vec<bool>,
    pub reset_flags: Vec<bool>,
    pub action_counts: Vec<usize>,
}

impl BatchedRolloutWorker {
    pub fn new(config: KernelConfig, states: Vec<KernelState>) -> Self {
        let histories = vec![Vec::new(); states.len()];
        let slot_configs = vec![config; states.len()];
        let rngs = vec![WorkerRng::Deterministic; states.len()];
        let initial_states = states.clone();
        let initial_histories = histories.clone();
        let initial_configs = slot_configs.clone();
        let episode_returns = vec![0.0; states.len()];
        let episode_lengths = vec![0; states.len()];
        Self {
            initial_states,
            initial_histories,
            initial_configs,
            reset_pool_states: None,
            reset_pool_histories: None,
            reset_pool_configs: None,
            next_reset_indices: vec![0; episode_returns.len()],
            states,
            histories,
            slot_configs,
            rngs,
            episode_returns,
            episode_lengths,
            action_feature_output: ActionFeatureOutput::DenseAndLegal,
            observation_output: ObservationOutput::V1AndV5,
            action_mask_output: ActionMaskOutput::Dense,
            terminal_observation_output: TerminalObservationOutput::Full,
            diagnostic_output: DiagnosticOutput::Full,
        }
    }

    pub fn from_snapshots(config: KernelConfig, snapshots: &[GoldenSnapshot]) -> Self {
        Self::from_snapshots_with_action_feature_output(
            config,
            snapshots,
            ActionFeatureOutput::DenseAndLegal,
        )
    }

    pub fn from_snapshots_with_action_feature_output(
        config: KernelConfig,
        snapshots: &[GoldenSnapshot],
        action_feature_output: ActionFeatureOutput,
    ) -> Self {
        Self::from_snapshots_with_outputs(
            config,
            snapshots,
            action_feature_output,
            ObservationOutput::V1AndV5,
            ActionMaskOutput::Dense,
            TerminalObservationOutput::Full,
            DiagnosticOutput::Full,
        )
    }

    pub fn from_snapshots_with_outputs(
        config: KernelConfig,
        snapshots: &[GoldenSnapshot],
        action_feature_output: ActionFeatureOutput,
        observation_output: ObservationOutput,
        action_mask_output: ActionMaskOutput,
        terminal_observation_output: TerminalObservationOutput,
        diagnostic_output: DiagnosticOutput,
    ) -> Self {
        let states: Vec<KernelState> = snapshots.iter().map(|s| s.state.clone()).collect();
        let histories: Vec<Vec<KernelHistoryEvent>> =
            snapshots.iter().map(|s| s.history_events.clone()).collect();
        let slot_configs = vec![config; states.len()];
        let rngs = vec![WorkerRng::Deterministic; states.len()];
        let initial_states = states.clone();
        let initial_histories = histories.clone();
        let initial_configs = slot_configs.clone();
        let episode_returns = vec![0.0; states.len()];
        let episode_lengths = vec![0; states.len()];
        Self {
            initial_states,
            initial_histories,
            initial_configs,
            reset_pool_states: None,
            reset_pool_histories: None,
            reset_pool_configs: None,
            next_reset_indices: vec![0; episode_returns.len()],
            states,
            histories,
            slot_configs,
            rngs,
            episode_returns,
            episode_lengths,
            action_feature_output,
            observation_output,
            action_mask_output,
            terminal_observation_output,
            diagnostic_output,
        }
    }

    pub fn from_snapshot_pool_with_action_feature_output(
        config: KernelConfig,
        snapshots: &[GoldenSnapshot],
        env_count: usize,
        action_feature_output: ActionFeatureOutput,
        cycle_resets: bool,
    ) -> Self {
        Self::from_snapshot_pool_with_outputs(
            config,
            snapshots,
            env_count,
            action_feature_output,
            ObservationOutput::V1AndV5,
            ActionMaskOutput::Dense,
            TerminalObservationOutput::Full,
            DiagnosticOutput::Full,
            cycle_resets,
        )
    }

    pub fn from_snapshot_pool_with_outputs(
        config: KernelConfig,
        snapshots: &[GoldenSnapshot],
        env_count: usize,
        action_feature_output: ActionFeatureOutput,
        observation_output: ObservationOutput,
        action_mask_output: ActionMaskOutput,
        terminal_observation_output: TerminalObservationOutput,
        diagnostic_output: DiagnosticOutput,
        cycle_resets: bool,
    ) -> Self {
        let env_snapshots = (0..env_count)
            .map(|idx| snapshots[idx % snapshots.len()].clone())
            .collect::<Vec<_>>();
        let mut worker = Self::from_snapshots_with_outputs(
            config,
            &env_snapshots,
            action_feature_output,
            observation_output,
            action_mask_output,
            terminal_observation_output,
            diagnostic_output,
        );
        if cycle_resets {
            worker.reset_pool_states = Some(snapshots.iter().map(|s| s.state.clone()).collect());
            worker.reset_pool_histories =
                Some(snapshots.iter().map(|s| s.history_events.clone()).collect());
            worker.reset_pool_configs = Some(vec![config; snapshots.len()]);
            worker.next_reset_indices = (0..env_count)
                .map(|idx| (idx + 1) % snapshots.len())
                .collect();
        }
        worker
    }

    pub fn from_snapshot_pool_with_trace_configs(
        config: KernelConfig,
        snapshots: &[(GoldenSnapshot, GoldenTraceConfig)],
        env_count: usize,
        action_feature_output: ActionFeatureOutput,
        observation_output: ObservationOutput,
        action_mask_output: ActionMaskOutput,
        terminal_observation_output: TerminalObservationOutput,
        diagnostic_output: DiagnosticOutput,
        cycle_resets: bool,
    ) -> Result<Self, String> {
        for (_, trace_config) in snapshots {
            let candidate = KernelConfig::from_trace_config(trace_config);
            validate_shared_config(config, candidate)?;
        }

        let env_entries = (0..env_count)
            .map(|idx| snapshots[idx % snapshots.len()].clone())
            .collect::<Vec<_>>();
        let states: Vec<KernelState> = env_entries
            .iter()
            .map(|(snapshot, _)| snapshot.state.clone())
            .collect();
        let histories: Vec<Vec<KernelHistoryEvent>> = env_entries
            .iter()
            .map(|(snapshot, _)| snapshot.history_events.clone())
            .collect();
        let slot_configs: Vec<KernelConfig> = env_entries
            .iter()
            .map(|(_, trace_config)| KernelConfig::from_trace_config(trace_config))
            .collect();
        let rngs: Vec<WorkerRng> = slot_configs
            .iter()
            .map(|_| WorkerRng::Deterministic)
            .collect();
        let initial_states = states.clone();
        let initial_histories = histories.clone();
        let initial_configs = slot_configs.clone();
        let episode_returns = vec![0.0; states.len()];
        let episode_lengths = vec![0; states.len()];
        let mut worker = Self {
            initial_states,
            initial_histories,
            initial_configs,
            reset_pool_states: None,
            reset_pool_histories: None,
            reset_pool_configs: None,
            next_reset_indices: vec![0; episode_returns.len()],
            states,
            histories,
            slot_configs,
            rngs,
            episode_returns,
            episode_lengths,
            action_feature_output,
            observation_output,
            action_mask_output,
            terminal_observation_output,
            diagnostic_output,
        };
        if cycle_resets {
            worker.reset_pool_states = Some(
                snapshots
                    .iter()
                    .map(|(snapshot, _)| snapshot.state.clone())
                    .collect(),
            );
            worker.reset_pool_histories = Some(
                snapshots
                    .iter()
                    .map(|(snapshot, _)| snapshot.history_events.clone())
                    .collect(),
            );
            worker.reset_pool_configs = Some(
                snapshots
                    .iter()
                    .map(|(_, trace_config)| KernelConfig::from_trace_config(trace_config))
                    .collect(),
            );
            worker.next_reset_indices = (0..env_count)
                .map(|idx| (idx + 1) % snapshots.len())
                .collect();
        }
        Ok(worker)
    }

    pub fn env_count(&self) -> usize {
        self.states.len()
    }

    pub fn current_actor_ids(&self) -> Vec<i32> {
        self.states
            .iter()
            .map(|state| state.current_turn_owner_id)
            .collect()
    }

    /// Per-env hero hp snapshot for the A4 live-self-play decisive-early-end
    /// predicate (BLOCK_A_PLAN.md A4, ``ppo_phaseA_config.is_decisive_state``).
    /// Returns a flat ``env_count * 4`` array laid out as
    /// ``[p1_hp, p1_max_hp, p2_hp, p2_max_hp]`` per env. Read-only view of the
    /// existing per-env ``KernelState`` — additive accessor, no behavior change.
    pub fn hero_hp(&self) -> Vec<i32> {
        let mut out = Vec::with_capacity(self.states.len() * 4);
        for state in &self.states {
            out.push(state.p1.hero.hp);
            out.push(state.p1.hero.max_hp);
            out.push(state.p2.hero.hp);
            out.push(state.p2.hero.max_hp);
        }
        out
    }

    pub fn select_rule_actions(&self, agent_codes: &[u32], salt: u64) -> Result<Vec<usize>, String> {
        if agent_codes.len() != self.states.len() {
            return Err(format!(
                "agent_codes length {} does not match env_count {}",
                agent_codes.len(),
                self.states.len()
            ));
        }

        let mut actions = Vec::with_capacity(self.states.len());
        for (idx, code) in agent_codes.iter().copied().enumerate() {
            let state = &self.states[idx];
            if state.status != "ongoing" {
                actions.push(0);
                continue;
            }
            let actor_id = state.current_turn_owner_id;
            let kernel = RolloutKernel::new(self.slot_configs[idx]);
            let legal_ids = kernel.legal_action_ids(state, actor_id);
            if legal_ids.is_empty() {
                return Err(format!("env slot {idx} has no legal actions"));
            }
            let selected = select_rule_action_for_state(code, state, actor_id, idx, salt, &legal_ids)?;
            actions.push(selected);
        }
        Ok(actions)
    }

    pub fn advance_rule_until_actor(
        &mut self,
        learner_actor_ids: &[i32],
        agent_codes: &[u32],
        max_actions_per_env: usize,
        salt: u64,
        auto_reset: bool,
    ) -> Result<RuleAdvanceOutput, String> {
        if learner_actor_ids.len() != self.states.len() {
            return Err(format!(
                "learner_actor_ids length {} does not match env_count {}",
                learner_actor_ids.len(),
                self.states.len()
            ));
        }
        if agent_codes.len() != self.states.len() {
            return Err(format!(
                "agent_codes length {} does not match env_count {}",
                agent_codes.len(),
                self.states.len()
            ));
        }
        if max_actions_per_env == 0 {
            return Err("max_actions_per_env must be positive".to_string());
        }

        let mut learner_rewards = vec![0.0_f32; self.states.len()];
        let mut terminated = vec![false; self.states.len()];
        let mut truncated = vec![false; self.states.len()];
        let mut reset_flags = vec![false; self.states.len()];
        let mut action_counts = vec![0_usize; self.states.len()];

        for idx in 0..self.states.len() {
            // Mirrors `ai/train_v2/rollout_worker.py::_auto_play_until_learner`
            // (rollout_worker.py:~243): the loop runs while the episode is
            // ongoing AND it is not the learner's turn. Truncation
            // (`turn_number > max_turns`) is independent of `terminated`
            // (kernel.rs:560-564) — when the opponent's step crosses
            // `max_turns`, `step.truncated` becomes true and we stop the
            // auto-advance so the caller can reset BEFORE the learner acts
            // (rollout_worker.py:~375), matching Python's
            // `while not terminated and not truncated`.
            while self.states[idx].status == "ongoing"
                && self.states[idx].current_turn_owner_id != learner_actor_ids[idx]
            {
                if action_counts[idx] >= max_actions_per_env {
                    return Err(format!(
                        "env slot {idx} exceeded max opponent actions {max_actions_per_env}"
                    ));
                }
                let actor_id = self.states[idx].current_turn_owner_id;
                let kernel = RolloutKernel::new(self.slot_configs[idx]);
                let legal_ids = kernel.legal_action_ids(&self.states[idx], actor_id);
                if legal_ids.is_empty() {
                    return Err(format!("env slot {idx} has no legal opponent actions"));
                }
                let action_id = select_rule_action_for_state(
                    agent_codes[idx],
                    &self.states[idx],
                    actor_id,
                    idx,
                    salt ^ mix_u64(action_counts[idx] as u64),
                    &legal_ids,
                )?;
                let mut draw_rng = DrawRng::live(&mut self.rngs[idx]);
                let step = kernel.apply_action(&self.states[idx], actor_id, action_id, false, &mut draw_rng)?;
                let event = build_history_event_v5(
                    &self.states[idx],
                    &step.state,
                    actor_id,
                    action_id,
                    action_type_for_id(action_id).to_string(),
                );

                let step_terminated = step.terminated;
                let step_truncated = step.truncated;
                self.states[idx] = step.state;
                self.episode_returns[idx] += step.reward;
                self.episode_lengths[idx] += 1;
                self.histories[idx].push(event);
                if self.histories[idx].len() > crate::v5::HISTORY_EVENTS {
                    let extra = self.histories[idx].len() - crate::v5::HISTORY_EVENTS;
                    self.histories[idx].drain(0..extra);
                }

                // The reward contract is perspective-dependent and not
                // zero-sum; use the kernel's exact other-player reward rather
                // than negating the acting opponent's reward.
                learner_rewards[idx] += step.counterparty_reward;
                action_counts[idx] += 1;
                if step_terminated {
                    terminated[idx] = true;
                    if auto_reset {
                        self.reset_slot(idx);
                        reset_flags[idx] = true;
                    }
                    break;
                }
                // Truncation is independent of termination (kernel.rs:560-564):
                // `turn_number > max_turns` does NOT flip `status`. We must NOT
                // auto-reset on truncation — `step_into` only resets on
                // `terminated` (worker.rs:797) and Python's
                // `_auto_play_until_learner` reports truncated without resetting
                // (the caller resets on the next "reset" cmd). Mirror that: set
                // the flag and stop the loop so the learner does not act in a
                // truncated state.
                if step_truncated {
                    truncated[idx] = true;
                    break;
                }
            }
        }

        Ok(RuleAdvanceOutput {
            learner_rewards,
            terminated,
            truncated,
            reset_flags,
            action_counts,
        })
    }

    pub fn reset_all(&mut self) {
        for idx in 0..self.states.len() {
            self.reset_slot(idx);
        }
    }

    /// Switch every slot's RNG to the deterministic (zero-byte) source.
    ///
    /// Intended for golden-fixture replay only: the frozen fixtures' draws
    /// are all `deck[0]`, and a zero-RNG makes `weighted_choice_idx` pick
    /// index 0 every time, so the recorded post-states reproduce exactly.
    /// The training path never calls this — it uses `use_chacha_rng`.
    pub fn use_deterministic_rng(&mut self) {
        for rng in &mut self.rngs {
            *rng = WorkerRng::Deterministic;
        }
    }

    /// Switch every slot's RNG to `ChaCha` seeded from that slot's
    /// `KernelConfig.seed`. Training calls this once at init to get real
    /// No-FIFO weighted draws; golden-fixture replay leaves the default
    /// `Deterministic` RNG so the frozen fixtures reproduce.
    pub fn use_chacha_rng(&mut self) {
        for idx in 0..self.rngs.len() {
            self.rngs[idx] = WorkerRng::cha_cha(self.slot_configs[idx].seed);
        }
    }

    pub fn reset_indices(&mut self, indices: &[usize]) -> Result<(), String> {
        for idx in indices {
            if *idx >= self.states.len() {
                return Err(format!(
                    "reset index {} out of bounds for env_count {}",
                    idx,
                    self.states.len()
                ));
            }
        }

        for idx in indices {
            self.reset_slot(*idx);
        }
        Ok(())
    }

    fn reset_slot(&mut self, idx: usize) {
        let pool_reset = self
            .reset_pool_states
            .as_ref()
            .zip(self.reset_pool_histories.as_ref())
            .zip(self.reset_pool_configs.as_ref())
            .map(|((states, histories), configs)| {
                let pool_idx = self.next_reset_indices[idx] % states.len();
                (
                    states[pool_idx].clone(),
                    histories[pool_idx].clone(),
                    configs[pool_idx],
                    (pool_idx + 1) % states.len(),
                )
            });
        if let Some((state, history, config, next_idx)) = pool_reset {
            self.states[idx] = state;
            self.histories[idx] = history;
            self.slot_configs[idx] = config;
            self.next_reset_indices[idx] = next_idx;
        } else {
            self.states[idx] = self.initial_states[idx].clone();
            self.histories[idx] = self.initial_histories[idx].clone();
            self.slot_configs[idx] = self.initial_configs[idx];
        }
        // Re-seed the slot RNG for the new episode. Deterministic mode is
        // preserved (golden-fixture replay); ChaCha is re-seeded from the
        // slot's config seed so a fresh episode starts a fresh stream.
        match self.rngs[idx] {
            WorkerRng::Deterministic => {}
            _ => self.rngs[idx] = WorkerRng::cha_cha(self.slot_configs[idx].seed),
        }
        self.episode_returns[idx] = 0.0;
        self.episode_lengths[idx] = 0;
    }

    pub fn encode_all(&self) -> BatchTensorOutput {
        let mut out = BatchTensorOutput::new(
            self.env_count(),
            self.action_feature_output,
            self.observation_output,
            self.action_mask_output,
            self.terminal_observation_output,
            self.diagnostic_output,
        );
        // Snapshot encoding is pure per slot and dominates live rollout time.
        // Keep the output assembly ordered and sequential, but compute those
        // independent snapshots in Rayon so the training process uses the CPU
        // budget selected by RAYON_NUM_THREADS (three threads in Block B).
        let snapshots: Vec<_> = (0..self.env_count())
            .into_par_iter()
            .map(|idx| {
                let state = &self.states[idx];
                let kernel = RolloutKernel::new(self.slot_configs[idx]);
                let snapshot = kernel.encode_snapshot_with_history(
                    state,
                    state.current_turn_owner_id,
                    &self.histories[idx],
                );
                (snapshot, self.episode_returns[idx], self.episode_lengths[idx], state.status != "ongoing")
            })
            .collect();
        for (snapshot, episode_return, episode_length, terminated) in snapshots {
            out.push_observation_v1(&snapshot.observation_v1);
            out.observation_v5
                .extend_from_slice(&snapshot.observation_v5);
            out.push_action_tensors(&snapshot.action_mask, &snapshot.action_features);
            out.mana_draw_legal.push(snapshot.mana_draw_legal);
            out.rewards.push(0.0);
            out.counterparty_rewards.push(0.0);
            out.terminated.push(terminated);
            out.truncated.push(false);
            out.push_reset_flag(false);
            out.push_empty_terminal_observation();
            out.push_episode_stats(episode_return, episode_length);
        }
        out
    }

    pub fn step(&mut self, action_ids: &[usize]) -> Result<BatchTensorOutput, String> {
        let flags = vec![false; action_ids.len()];
        self.step_with_mana_draw(action_ids, &flags, false)
    }

    pub fn step_auto_reset(&mut self, action_ids: &[usize]) -> Result<BatchTensorOutput, String> {
        let flags = vec![false; action_ids.len()];
        self.step_with_mana_draw(action_ids, &flags, true)
    }

    /// Step the batch with a parallel mana_draw flag per env (Phase 2: MD-3,
    /// AC-FFI-1/MD-FFI). `mana_draw_flags` must be the same length as
    /// `action_ids`; when `mana_draw_flags[idx]` is true, env `idx` applies a
    /// mana-draw for its current actor instead of decoding `action_ids[idx]`.
    pub fn step_with_mana_draw(
        &mut self,
        action_ids: &[usize],
        mana_draw_flags: &[bool],
        auto_reset: bool,
    ) -> Result<BatchTensorOutput, String> {
        let mut out = BatchTensorOutput::new(
            self.env_count(),
            self.action_feature_output,
            self.observation_output,
            self.action_mask_output,
            self.terminal_observation_output,
            self.diagnostic_output,
        );
        self.step_into(action_ids, mana_draw_flags, auto_reset, &mut out)?;
        Ok(out)
    }

    fn step_into(
        &mut self,
        action_ids: &[usize],
        mana_draw_flags: &[bool],
        auto_reset: bool,
        out: &mut BatchTensorOutput,
    ) -> Result<(), String> {
        if action_ids.len() != self.states.len() {
            return Err(format!(
                "action_ids length {} does not match env_count {}",
                action_ids.len(),
                self.states.len()
            ));
        }
        if mana_draw_flags.len() != self.states.len() {
            return Err(format!(
                "mana_draw_flags length {} does not match env_count {}",
                mana_draw_flags.len(),
                self.states.len()
            ));
        }

        for (idx, action_id) in action_ids.iter().copied().enumerate() {
            let mana_draw_flag = mana_draw_flags[idx];
            let actor_id = self.states[idx].current_turn_owner_id;
            let kernel = RolloutKernel::new(self.slot_configs[idx]);
            let mut draw_rng = DrawRng::live(&mut self.rngs[idx]);
            let step = kernel.apply_action(
                &self.states[idx],
                actor_id,
                action_id,
                mana_draw_flag,
                &mut draw_rng,
            ).map_err(|err| {
                format!(
                    "env {idx} action_id={action_id} mana_draw={mana_draw_flag} actor={actor_id}: {err}"
                )
            })?;
            let action_type = if mana_draw_flag {
                "mana_draw".to_string()
            } else {
                action_type_for_id(action_id).to_string()
            };
            let event = build_history_event_v5(
                &self.states[idx],
                &step.state,
                actor_id,
                if mana_draw_flag { MAX_CANDIDATE_ACTIONS } else { action_id },
                action_type,
            );

            let terminated = step.terminated;
            self.states[idx] = step.state;
            self.episode_returns[idx] += step.reward;
            self.episode_lengths[idx] += 1;
            let episode_return = self.episode_returns[idx];
            let episode_length = self.episode_lengths[idx];
            self.histories[idx].push(event);
            if self.histories[idx].len() > crate::v5::HISTORY_EVENTS {
                let extra = self.histories[idx].len() - crate::v5::HISTORY_EVENTS;
                self.histories[idx].drain(0..extra);
            }

            if auto_reset && terminated {
                let terminal_snapshot = kernel.encode_snapshot_with_history(
                    &self.states[idx],
                    self.states[idx].current_turn_owner_id,
                    &self.histories[idx],
                );
                out.push_terminal_observation(
                    &terminal_snapshot.observation_v1,
                    &terminal_snapshot.observation_v5,
                );
            } else {
                out.push_empty_terminal_observation();
            }

            if auto_reset && terminated {
                self.reset_slot(idx);
            }

            let kernel = RolloutKernel::new(self.slot_configs[idx]);
            let snapshot = kernel.encode_snapshot_with_history(
                &self.states[idx],
                self.states[idx].current_turn_owner_id,
                &self.histories[idx],
            );
            out.push_observation_v1(&snapshot.observation_v1);
            out.observation_v5
                .extend_from_slice(&snapshot.observation_v5);
            out.push_action_tensors(&snapshot.action_mask, &snapshot.action_features);
            out.mana_draw_legal.push(snapshot.mana_draw_legal);
            out.rewards.push(step.reward);
            out.counterparty_rewards.push(step.counterparty_reward);
            out.terminated.push(terminated);
            out.truncated.push(step.truncated);
            out.push_reset_flag(auto_reset && terminated);
            out.push_episode_stats(episode_return, episode_length);
        }
        Ok(())
    }

    fn step_pre_action_into(
        &mut self,
        action_ids: &[usize],
        auto_reset: bool,
        out: &mut BatchTensorOutput,
    ) -> Result<(), String> {
        if action_ids.len() != self.states.len() {
            return Err(format!(
                "action_ids length {} does not match env_count {}",
                action_ids.len(),
                self.states.len()
            ));
        }

        for (idx, action_id) in action_ids.iter().copied().enumerate() {
            let kernel = RolloutKernel::new(self.slot_configs[idx]);
            let pre_snapshot = kernel.encode_snapshot_with_history(
                &self.states[idx],
                self.states[idx].current_turn_owner_id,
                &self.histories[idx],
            );
            out.push_observation_v1(&pre_snapshot.observation_v1);
            out.observation_v5
                .extend_from_slice(&pre_snapshot.observation_v5);
            out.push_action_tensors_with_selected_action(
                &pre_snapshot.action_mask,
                &pre_snapshot.action_features,
                action_id,
            )?;
            out.mana_draw_legal.push(pre_snapshot.mana_draw_legal);

            let actor_id = self.states[idx].current_turn_owner_id;
            let mut draw_rng = DrawRng::live(&mut self.rngs[idx]);
            let step = kernel.apply_action(&self.states[idx], actor_id, action_id, false, &mut draw_rng)?;
            let event = build_history_event_v5(
                &self.states[idx],
                &step.state,
                actor_id,
                action_id,
                action_type_for_id(action_id).to_string(),
            );

            let terminated = step.terminated;
            self.states[idx] = step.state;
            self.episode_returns[idx] += step.reward;
            self.episode_lengths[idx] += 1;
            let episode_return = self.episode_returns[idx];
            let episode_length = self.episode_lengths[idx];
            self.histories[idx].push(event);
            if self.histories[idx].len() > crate::v5::HISTORY_EVENTS {
                let extra = self.histories[idx].len() - crate::v5::HISTORY_EVENTS;
                self.histories[idx].drain(0..extra);
            }

            if auto_reset && terminated {
                let terminal_snapshot = kernel.encode_snapshot_with_history(
                    &self.states[idx],
                    self.states[idx].current_turn_owner_id,
                    &self.histories[idx],
                );
                out.push_terminal_observation(
                    &terminal_snapshot.observation_v1,
                    &terminal_snapshot.observation_v5,
                );
            } else {
                out.push_empty_terminal_observation();
            }

            if auto_reset && terminated {
                self.reset_slot(idx);
            }

            out.rewards.push(step.reward);
            out.counterparty_rewards.push(step.counterparty_reward);
            out.terminated.push(terminated);
            out.truncated.push(step.truncated);
            out.push_reset_flag(auto_reset && terminated);
            out.push_episode_stats(episode_return, episode_length);
        }
        Ok(())
    }

    pub fn rollout_action_tape(
        &mut self,
        action_ids: &[usize],
        steps: usize,
        auto_reset: bool,
    ) -> Result<BatchTensorOutput, String> {
        if steps == 0 {
            return Err("steps must be positive".to_string());
        }
        let env_count = self.states.len();
        let Some(expected_actions) = steps.checked_mul(env_count) else {
            return Err("action tape length overflow".to_string());
        };
        if action_ids.len() != expected_actions {
            return Err(format!(
                "action_ids length {} does not match steps {} * env_count {}",
                action_ids.len(),
                steps,
                env_count
            ));
        }

        let mut out = BatchTensorOutput::new(
            expected_actions,
            self.action_feature_output,
            self.observation_output,
            self.action_mask_output,
            self.terminal_observation_output,
            self.diagnostic_output,
        );
        for actions in action_ids.chunks_exact(env_count) {
            let flags = vec![false; env_count];
            self.step_into(actions, &flags, auto_reset, &mut out)?;
        }
        Ok(out)
    }

    pub fn rollout_broadcast_action_ids(
        &mut self,
        step_action_ids: &[usize],
        auto_reset: bool,
    ) -> Result<BatchTensorOutput, String> {
        if step_action_ids.is_empty() {
            return Err("step_action_ids must be non-empty".to_string());
        }
        let env_count = self.states.len();
        let Some(expected_rows) = step_action_ids.len().checked_mul(env_count) else {
            return Err("broadcast action tape length overflow".to_string());
        };

        let mut out = BatchTensorOutput::new(
            expected_rows,
            self.action_feature_output,
            self.observation_output,
            self.action_mask_output,
            self.terminal_observation_output,
            self.diagnostic_output,
        );
        let mut broadcast_actions = vec![0_usize; env_count];
        let flags = vec![false; env_count];
        for action_id in step_action_ids.iter().copied() {
            broadcast_actions.fill(action_id);
            self.step_into(&broadcast_actions, &flags, auto_reset, &mut out)?;
        }
        Ok(out)
    }

    pub fn rollout_action_tape_pre_step(
        &mut self,
        action_ids: &[usize],
        steps: usize,
        auto_reset: bool,
    ) -> Result<BatchTensorOutput, String> {
        if steps == 0 {
            return Err("steps must be positive".to_string());
        }
        let env_count = self.states.len();
        let Some(expected_actions) = steps.checked_mul(env_count) else {
            return Err("action tape length overflow".to_string());
        };
        if action_ids.len() != expected_actions {
            return Err(format!(
                "action_ids length {} does not match steps {} * env_count {}",
                action_ids.len(),
                steps,
                env_count
            ));
        }

        let mut out = BatchTensorOutput::new(
            expected_actions,
            self.action_feature_output,
            self.observation_output,
            self.action_mask_output,
            self.terminal_observation_output,
            self.diagnostic_output,
        );
        for actions in action_ids.chunks_exact(env_count) {
            self.step_pre_action_into(actions, auto_reset, &mut out)?;
        }
        Ok(out)
    }

    pub fn rollout_broadcast_action_ids_pre_step(
        &mut self,
        step_action_ids: &[usize],
        auto_reset: bool,
    ) -> Result<BatchTensorOutput, String> {
        if step_action_ids.is_empty() {
            return Err("step_action_ids must be non-empty".to_string());
        }
        let env_count = self.states.len();
        let Some(expected_rows) = step_action_ids.len().checked_mul(env_count) else {
            return Err("broadcast action tape length overflow".to_string());
        };

        let mut out = BatchTensorOutput::new(
            expected_rows,
            self.action_feature_output,
            self.observation_output,
            self.action_mask_output,
            self.terminal_observation_output,
            self.diagnostic_output,
        );
        let mut broadcast_actions = vec![0_usize; env_count];
        for action_id in step_action_ids.iter().copied() {
            broadcast_actions.fill(action_id);
            self.step_pre_action_into(&broadcast_actions, auto_reset, &mut out)?;
        }
        Ok(out)
    }
}

impl BatchTensorOutput {
    fn new(
        env_count: usize,
        action_feature_output: ActionFeatureOutput,
        observation_output: ObservationOutput,
        action_mask_output: ActionMaskOutput,
        terminal_observation_output: TerminalObservationOutput,
        diagnostic_output: DiagnosticOutput,
    ) -> Self {
        let dense_feature_capacity = if action_feature_output.include_dense() {
            env_count * 601 * ACTION_FEATURE_DIM_V1
        } else {
            0
        };
        let v1_capacity = if observation_output.include_v1() {
            env_count * OBS_DIM_V1
        } else {
            0
        };
        let v5_capacity = env_count * OBS_DIM_V5;
        let terminal_v1_capacity = if terminal_observation_output.include_terminal_observations()
            && observation_output.include_v1()
        {
            env_count * OBS_DIM_V1
        } else {
            0
        };
        let terminal_v5_capacity = if terminal_observation_output.include_terminal_observations() {
            env_count * OBS_DIM_V5
        } else {
            0
        };
        let mask_capacity = if action_mask_output.include_dense() {
            env_count * 601
        } else {
            0
        };
        let diagnostic_capacity = if diagnostic_output.include_diagnostics() {
            env_count
        } else {
            0
        };
        Self {
            env_count,
            observation_v1: Vec::with_capacity(v1_capacity),
            observation_v5: Vec::with_capacity(v5_capacity),
            action_mask: Vec::with_capacity(mask_capacity),
            action_features: Vec::with_capacity(dense_feature_capacity),
            legal_action_counts: Vec::with_capacity(env_count),
            legal_action_offsets: Vec::with_capacity(env_count),
            legal_action_ids: Vec::new(),
            legal_action_features: Vec::new(),
            selected_local_indices: Vec::with_capacity(env_count),
            rewards: Vec::with_capacity(env_count),
            counterparty_rewards: Vec::with_capacity(env_count),
            terminated: Vec::with_capacity(env_count),
            truncated: Vec::with_capacity(env_count),
            reset_flags: Vec::with_capacity(diagnostic_capacity),
            terminal_observation_v1: Vec::with_capacity(terminal_v1_capacity),
            terminal_observation_v5: Vec::with_capacity(terminal_v5_capacity),
            terminal_observation_valid: Vec::with_capacity(diagnostic_capacity),
            episode_returns: Vec::with_capacity(diagnostic_capacity),
            episode_lengths: Vec::with_capacity(diagnostic_capacity),
            mana_draw_legal: Vec::with_capacity(env_count),
        }
    }

    fn push_observation_v1(&mut self, observation_v1: &[f32]) {
        if self.observation_v1.capacity() > 0 {
            self.observation_v1.extend_from_slice(observation_v1);
        }
    }

    fn push_action_tensors(&mut self, mask: &[f32], features: &[f32]) {
        self.push_action_tensors_impl(mask, features, None)
            .expect("push_action_tensors without a selected action cannot fail");
    }

    fn push_action_tensors_with_selected_action(
        &mut self,
        mask: &[f32],
        features: &[f32],
        action_id: usize,
    ) -> Result<(), String> {
        self.push_action_tensors_impl(mask, features, Some(action_id))
    }

    fn push_action_tensors_impl(
        &mut self,
        mask: &[f32],
        features: &[f32],
        selected_action_id: Option<usize>,
    ) -> Result<(), String> {
        if self.action_mask.capacity() > 0 {
            self.action_mask.extend_from_slice(mask);
        }
        if self.action_features.capacity() > 0 {
            self.action_features.extend_from_slice(features);
        }

        let mut legal_count = 0_usize;
        let mut selected_local: Option<i32> = None;
        self.legal_action_offsets.push(self.legal_action_ids.len());
        for (action_id, value) in mask.iter().copied().enumerate() {
            if value > 0.0 {
                if selected_action_id.is_some_and(|selected| selected == action_id) {
                    selected_local = Some(legal_count as i32);
                }
                legal_count += 1;
                self.legal_action_ids.push(action_id);
                let start = action_id * ACTION_FEATURE_DIM_V1;
                let end = start + ACTION_FEATURE_DIM_V1;
                self.legal_action_features
                    .extend_from_slice(&features[start..end]);
            }
        }
        self.legal_action_counts.push(legal_count);

        if let Some(action_id) = selected_action_id {
            let Some(local_idx) = selected_local else {
                return Err(format!("action {action_id} is absent from legal ids"));
            };
            self.selected_local_indices.push(local_idx);
        }
        Ok(())
    }

    fn push_reset_flag(&mut self, reset: bool) {
        if self.reset_flags.capacity() > 0 {
            self.reset_flags.push(reset);
        }
    }

    fn push_episode_stats(&mut self, episode_return: f32, episode_length: usize) {
        if self.episode_returns.capacity() > 0 {
            self.episode_returns.push(episode_return);
            self.episode_lengths.push(episode_length);
        }
    }

    fn push_empty_terminal_observation(&mut self) {
        if self.terminal_observation_v1.capacity() > 0 {
            self.terminal_observation_v1
                .extend(std::iter::repeat(0.0_f32).take(OBS_DIM_V1));
        }
        if self.terminal_observation_v5.capacity() > 0 {
            self.terminal_observation_v5
                .extend(std::iter::repeat(0.0_f32).take(OBS_DIM_V5));
        }
        if self.terminal_observation_valid.capacity() > 0 {
            self.terminal_observation_valid.push(false);
        }
    }

    fn push_terminal_observation(&mut self, observation_v1: &[f32], observation_v5: &[f32]) {
        if self.terminal_observation_v1.capacity() > 0 {
            self.terminal_observation_v1
                .extend_from_slice(observation_v1);
        }
        if self.terminal_observation_v5.capacity() > 0 {
            self.terminal_observation_v5
                .extend_from_slice(observation_v5);
        }
        if self.terminal_observation_valid.capacity() > 0 {
            self.terminal_observation_valid.push(true);
        }
    }
}

fn exploit_agent_kind_from_code(code: u32) -> Option<ExploitAgentKind> {
    match code {
        1 => Some(ExploitAgentKind::FaceRush),
        2 => Some(ExploitAgentKind::BoardControl),
        3 => Some(ExploitAgentKind::GreedyTrade),
        4 => Some(ExploitAgentKind::Stall),
        5 => Some(ExploitAgentKind::PunishEmptyBoard),
        6 => Some(ExploitAgentKind::AntiDrawGreed),
        7 => Some(ExploitAgentKind::AntiHandLeakOverfit),
        _ => None,
    }
}

fn select_deterministic_legal_random_action(
    state: &KernelState,
    slot_idx: usize,
    salt: u64,
    legal_ids: &[usize],
) -> Option<usize> {
    if legal_ids.is_empty() {
        return None;
    }
    let mut hash = salt ^ mix_u64(slot_idx as u64);
    hash ^= mix_u64(state.turn_number as i64 as u64);
    hash ^= mix_u64(state.current_turn_owner_id as i64 as u64);
    hash ^= mix_u64(state.p1.hero.hp as i64 as u64);
    hash ^= mix_u64((state.p2.hero.hp as i64 as u64) << 1);
    hash ^= mix_u64(((state.p1.hand.len() as u64) << 8) | state.p2.hand.len() as u64);
    hash ^= mix_u64(((state.p1.board.len() as u64) << 16) | state.p2.board.len() as u64);
    hash ^= mix_u64(((state.p1.deck.len() as u64) << 24) | state.p2.deck.len() as u64);
    Some(legal_ids[(hash as usize) % legal_ids.len()])
}

fn select_rule_action_for_state(
    code: u32,
    state: &KernelState,
    actor_id: i32,
    slot_idx: usize,
    salt: u64,
    legal_ids: &[usize],
) -> Result<usize, String> {
    let selected = match code {
        0 => select_deterministic_legal_random_action(state, slot_idx, salt, legal_ids),
        _ => {
            let Some(kind) = exploit_agent_kind_from_code(code) else {
                return Err(format!("unknown rule agent code {code}"));
            };
            select_exploit_action_for_state(kind, state, actor_id, legal_ids)
        }
    }
    .unwrap_or(legal_ids[0]);
    Ok(selected)
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            env_count: 1,
            max_steps_per_episode: 500,
            action_features_float16: false,
            env: TrainEnvConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActionFeatureOutput, ActionMaskOutput, BatchTensorOutput, DiagnosticOutput,
        ObservationOutput, TerminalObservationOutput,
    };
    use crate::action_codec::ATTACK_BASE;
    use crate::kernel::{KernelCard, KernelConfig, KernelPlayer, KernelState};
    use crate::ACTION_FEATURE_DIM_V1;

    fn card(id: i32, cost: i32, atk: i32, hp: i32) -> KernelCard {
        KernelCard {
            card_id: id,
            card_type: "warrior".to_string(),
            mana_cost: cost,
            attack: atk,
            hp,
            max_hp: hp,
            ..Default::default()
        }
    }

    fn player_with(deck: Vec<KernelCard>) -> KernelPlayer {
        KernelPlayer {
            user_id: 1,
            mana: 10,
            max_mana: 10,
            hero: KernelCard {
                card_id: 0,
                card_type: "hero".to_string(),
                mana_cost: 0,
                attack: 0,
                hp: 30,
                max_hp: 30,
                ..Default::default()
            },
            hand: Vec::new(),
            board: Vec::new(),
            deck,
            graveyard: Vec::new(),
            trophies: 0,
            mana_draw_count_this_turn: 0,
        }
    }

    #[test]
    fn push_action_tensors_records_selected_local_during_legal_encoding() {
        let mut output = BatchTensorOutput::new(
            1,
            ActionFeatureOutput::LegalOnly,
            ObservationOutput::V5Only,
            ActionMaskOutput::LegalOnly,
            TerminalObservationOutput::None,
            DiagnosticOutput::None,
        );
        let mut mask = vec![0.0_f32; 601];
        mask[3] = 1.0;
        mask[7] = 1.0;
        mask[10] = 1.0;
        let mut features = vec![0.0_f32; 601 * ACTION_FEATURE_DIM_V1];
        features[3 * ACTION_FEATURE_DIM_V1] = 3.0;
        features[7 * ACTION_FEATURE_DIM_V1] = 7.0;
        features[10 * ACTION_FEATURE_DIM_V1] = 10.0;

        output
            .push_action_tensors_with_selected_action(&mask, &features, 7)
            .expect("selected action is legal");

        assert_eq!(output.legal_action_counts, vec![3]);
        assert_eq!(output.legal_action_offsets, vec![0]);
        assert_eq!(output.legal_action_ids, vec![3, 7, 10]);
        assert_eq!(output.selected_local_indices, vec![1]);
        assert_eq!(
            output.legal_action_features.len(),
            3 * ACTION_FEATURE_DIM_V1
        );
        assert_eq!(output.legal_action_features[0], 3.0);
        assert_eq!(output.legal_action_features[ACTION_FEATURE_DIM_V1], 7.0);
        assert_eq!(
            output.legal_action_features[2 * ACTION_FEATURE_DIM_V1],
            10.0
        );
    }

    #[test]
    fn worker_step_threads_truncated_flag_into_batch_output() {
        // Phase 7 (WD-2): the batched rollout worker must surface the
        // per-step `truncated` flag (turn_number > max_turns) into
        // `BatchTensorOutput.truncated`, independent of `terminated`. With
        // max_turns=4 and a pre-step turn_number of 4, an end_turn (action_id
        // 0) advances to turn 5 → truncated=true, terminated=false (status
        // still "ongoing"). The worker steps a single-env batch and the
        // output's truncated vec must reflect this.
        use super::BatchedRolloutWorker;
        use crate::kernel::{action_type_for_id, RolloutKernel};
        use crate::worker::WorkerRng;
        use crate::kernel::DrawRng;

        let mut cfg = KernelConfig::default();
        cfg.max_turns = 4;
        let mut p1 = player_with(Vec::new());
        p1.user_id = 1;
        let mut p2 = player_with(vec![card(12, 1, 1, 1)]);
        p2.user_id = 2;
        let state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 4,
            status: "ongoing".to_string(),
            p1,
            p2,
            ..Default::default()
        };
        // Build the worker from a one-env snapshot of the state. The kernel
        // is only used to confirm the action_id is a legal end_turn.
        let kernel = RolloutKernel::new(cfg);
        let mask = kernel.legal_action_ids(&state, 1);
        let &end_turn_id = mask
            .iter()
            .find(|&&id| action_type_for_id(id) == "end_turn")
            .expect("end_turn is legal");
        let mut worker = BatchedRolloutWorker::new(cfg, vec![state]);
        // encode_all seeds the truncated vec with false (no step taken yet).
        let initial = worker.encode_all();
        assert_eq!(initial.truncated, vec![false]);
        let mut det_rng = WorkerRng::Deterministic;
        // Step the batch with the end_turn action.
        let out = worker.step(&[end_turn_id]).expect("step applies");
        assert_eq!(out.terminated, vec![false], "status still ongoing");
        assert_eq!(out.truncated, vec![true], "turn 5 > max_turns 4 → truncated");
    }

    #[test]
    fn advance_rule_until_actor_propagates_truncated_when_opponent_crosses_max_turns() {
        // Phase 7 follow-up: the opponent-auto-advance path must surface
        // `truncated` (turn_number > max_turns) just like the learner-step
        // path (worker.rs:814). With max_turns=4 and the opponent to act at
        // turn_number=4, the opponent's only legal action is end_turn
        // (empty hand + empty board → 601-mask is just action_id 0). end_turn
        // advances turn_number to 5 > 4 → step.truncated=true, terminated=false
        // (status still "ongoing"). `RuleAdvanceOutput.truncated` must reflect
        // this so the caller resets on terminated-or-truncated BEFORE the
        // learner acts (mirrors rollout_worker.py:~375). Previously the advance
        // path had no `truncated` field and `encode_all` (worker.rs:684) seeded
        // all-false, hiding the truncation for one step.
        use super::BatchedRolloutWorker;

        let mut cfg = KernelConfig::default();
        cfg.max_turns = 4;
        let mut p1 = player_with(vec![card(12, 1, 1, 1)]);
        p1.user_id = 1;
        let mut p2 = player_with(vec![card(12, 1, 1, 1)]);
        p2.user_id = 2;
        // Empty hand + board for both → the only legal 601-candidate action
        // for the turn owner is end_turn (action_id 0); mana_draw is a separate
        // head not exercised by advance_rule_until_actor (mana_draw_flag=false).
        p1.hand = Vec::new();
        p1.board = Vec::new();
        p2.hand = Vec::new();
        p2.board = Vec::new();
        let state = KernelState {
            current_turn_owner_id: 2,
            turn_number: 4,
            status: "ongoing".to_string(),
            p1,
            p2,
            ..Default::default()
        };
        let mut worker = BatchedRolloutWorker::new(cfg, vec![state]);
        worker.use_deterministic_rng();
        // learner is player 1; opponent (player 2) acts first.
        let out = worker
            .advance_rule_until_actor(&[1], &[0], 64, 0, false)
            .expect("advance applies");
        assert_eq!(out.terminated, vec![false], "status still ongoing after end_turn");
        assert_eq!(
            out.truncated, vec![true],
            "opponent end_turn → turn 5 > max_turns 4 → truncated must propagate"
        );
        assert_eq!(out.action_counts, vec![1], "exactly one opponent action");
        assert_eq!(out.reset_flags, vec![false], "no auto-reset (auto_reset=false)");
    }

    #[test]
    fn worker_step_exposes_exact_counterparty_reward() {
        use super::BatchedRolloutWorker;

        let cfg = KernelConfig::default();
        let mut p1 = player_with(Vec::new());
        p1.user_id = 1;
        let mut attacker = card(55, 0, 5, 5);
        attacker.is_ready = true;
        p1.board = vec![attacker];
        let mut p2 = player_with(Vec::new());
        p2.user_id = 2;
        let state = KernelState {
            current_turn_owner_id: 1,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1,
            p2,
            ..Default::default()
        };
        let mut worker = BatchedRolloutWorker::new(cfg, vec![state]);
        let out = worker
            .step(&[ATTACK_BASE + 7])
            .expect("face attack applies");
        assert!((out.rewards[0] - 0.10).abs() < 1.0e-6);
        assert!((out.counterparty_rewards[0] - -0.05).abs() < 1.0e-6);
    }

    #[test]
    fn advance_rule_until_actor_uses_direct_learner_perspective_reward() {
        use super::BatchedRolloutWorker;

        let cfg = KernelConfig::default();
        let mut p1 = player_with(Vec::new());
        p1.user_id = 1;
        let mut p2 = player_with(Vec::new());
        p2.user_id = 2;
        let mut attacker = card(55, 0, 5, 5);
        attacker.is_ready = true;
        p2.board = vec![attacker];
        let state = KernelState {
            current_turn_owner_id: 2,
            turn_number: 1,
            status: "ongoing".to_string(),
            p1,
            p2,
            ..Default::default()
        };
        let mut worker = BatchedRolloutWorker::new(cfg, vec![state]);
        worker.use_deterministic_rng();
        let out = worker
            .advance_rule_until_actor(&[1], &[1], 64, 0, false)
            .expect("face-rush opponent advances to learner");

        assert_eq!(out.action_counts, vec![2], "face attack followed by end turn");
        assert!(
            (out.learner_rewards[0] - -0.05).abs() < 1.0e-6,
            "received damage is -0.01/hp for learner, not negated +0.02/hp actor reward"
        );
    }
}
