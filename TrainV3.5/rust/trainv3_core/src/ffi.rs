//! FFI boundary notes.
//!
//! The intended bridge is coarse-grained:
//! Python sends seeds, normalized catalog/decks, config flags, and batched action ids.
//! Rust owns episode state and returns contiguous tensor buffers plus compact summaries.
//!
//! Avoid per-step Python dataclass, UUID, or `BaseAction` transfer. Those belong at
//! parity/debug boundaries only.

use std::ptr;
use std::slice;

use half::f16;

use crate::kernel::{GoldenTrace, KernelConfig};
use crate::ppo::{
    compute_gae_returns, normalize_legal_offsets, pack_legal_action_rows, pad_legal_actions,
    prepare_ppo_batch, repeat_row_indices, select_compact_argmax_actions,
    select_dense_argmax_actions, select_local_indices, select_padded_argmax_actions,
};
use crate::worker::{
    ActionFeatureOutput, ActionMaskOutput, BatchTensorOutput, BatchedRolloutWorker,
    DiagnosticOutput, ObservationOutput, TerminalObservationOutput,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    Float32,
    Float16,
}

pub struct FfiWorker {
    worker: BatchedRolloutWorker,
    last: BatchTensorOutput,
    action_features_dtype: TensorDType,
    action_features_f16: Vec<u16>,
    legal_action_features_f16: Vec<u16>,
    terminated_u8: Vec<u8>,
    truncated_u8: Vec<u8>,
    reset_flags_u8: Vec<u8>,
    terminal_observation_valid_u8: Vec<u8>,
    mana_draw_legal_u8: Vec<u8>,
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_compute_gae(
    rewards_ptr: *const f32,
    values_ptr: *const f32,
    terminated_ptr: *const u8,
    truncated_ptr: *const u8,
    bootstrap_values_ptr: *const f32,
    steps: usize,
    env_count: usize,
    gamma: f32,
    gae_lambda: f32,
    normalize_advantages: u8,
    advantages_ptr: *mut f32,
    returns_ptr: *mut f32,
) -> i32 {
    if steps == 0 || env_count == 0 {
        return -1;
    }
    let Some(row_count) = steps.checked_mul(env_count) else {
        return -2;
    };
    if rewards_ptr.is_null()
        || values_ptr.is_null()
        || terminated_ptr.is_null()
        || advantages_ptr.is_null()
        || returns_ptr.is_null()
    {
        return -3;
    }
    let normalize = match normalize_advantages {
        0 => false,
        1 => true,
        _ => return -4,
    };

    let rewards = unsafe { slice::from_raw_parts(rewards_ptr, row_count) };
    let values = unsafe { slice::from_raw_parts(values_ptr, row_count) };
    let terminated = unsafe { slice::from_raw_parts(terminated_ptr, row_count) };
    let truncated = if truncated_ptr.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts(truncated_ptr, row_count) })
    };
    let bootstrap_values = if bootstrap_values_ptr.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts(bootstrap_values_ptr, env_count) })
    };
    let advantages_out = unsafe { slice::from_raw_parts_mut(advantages_ptr, row_count) };
    let returns_out = unsafe { slice::from_raw_parts_mut(returns_ptr, row_count) };

    match compute_gae_returns(
        rewards,
        values,
        terminated,
        truncated,
        bootstrap_values,
        steps,
        env_count,
        gamma,
        gae_lambda,
        normalize,
        advantages_out,
        returns_out,
    ) {
        Ok(()) => 0,
        Err(_) => -5,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_select_local_indices(
    actions_ptr: *const usize,
    legal_action_counts_ptr: *const usize,
    legal_action_offsets_ptr: *const usize,
    legal_action_ids_ptr: *const usize,
    legal_action_ids_len: usize,
    row_count: usize,
    selected_ptr: *mut i32,
) -> i32 {
    if row_count == 0 {
        return -1;
    }
    if actions_ptr.is_null()
        || legal_action_counts_ptr.is_null()
        || legal_action_offsets_ptr.is_null()
        || legal_action_ids_ptr.is_null()
        || selected_ptr.is_null()
    {
        return -2;
    }

    let actions = unsafe { slice::from_raw_parts(actions_ptr, row_count) };
    let counts = unsafe { slice::from_raw_parts(legal_action_counts_ptr, row_count) };
    let offsets = unsafe { slice::from_raw_parts(legal_action_offsets_ptr, row_count) };
    let legal_ids = unsafe { slice::from_raw_parts(legal_action_ids_ptr, legal_action_ids_len) };
    let selected_out = unsafe { slice::from_raw_parts_mut(selected_ptr, row_count) };

    match select_local_indices(actions, counts, offsets, legal_ids, selected_out) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_prepare_ppo_batch(
    rewards_ptr: *const f32,
    values_ptr: *const f32,
    terminated_ptr: *const u8,
    truncated_ptr: *const u8,
    bootstrap_values_ptr: *const f32,
    actions_ptr: *const usize,
    legal_action_counts_ptr: *const usize,
    legal_action_offsets_ptr: *const usize,
    legal_action_ids_ptr: *const usize,
    legal_action_ids_len: usize,
    steps: usize,
    env_count: usize,
    gamma: f32,
    gae_lambda: f32,
    normalize_advantages: u8,
    advantages_ptr: *mut f32,
    returns_ptr: *mut f32,
    selected_local_ptr: *mut i32,
) -> i32 {
    if steps == 0 || env_count == 0 {
        return -1;
    }
    let Some(row_count) = steps.checked_mul(env_count) else {
        return -2;
    };
    if rewards_ptr.is_null()
        || values_ptr.is_null()
        || terminated_ptr.is_null()
        || actions_ptr.is_null()
        || legal_action_counts_ptr.is_null()
        || legal_action_offsets_ptr.is_null()
        || legal_action_ids_ptr.is_null()
        || advantages_ptr.is_null()
        || returns_ptr.is_null()
        || selected_local_ptr.is_null()
    {
        return -3;
    }
    let normalize = match normalize_advantages {
        0 => false,
        1 => true,
        _ => return -4,
    };

    let rewards = unsafe { slice::from_raw_parts(rewards_ptr, row_count) };
    let values = unsafe { slice::from_raw_parts(values_ptr, row_count) };
    let terminated = unsafe { slice::from_raw_parts(terminated_ptr, row_count) };
    let truncated = if truncated_ptr.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts(truncated_ptr, row_count) })
    };
    let bootstrap_values = if bootstrap_values_ptr.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts(bootstrap_values_ptr, env_count) })
    };
    let actions = unsafe { slice::from_raw_parts(actions_ptr, row_count) };
    let counts = unsafe { slice::from_raw_parts(legal_action_counts_ptr, row_count) };
    let offsets = unsafe { slice::from_raw_parts(legal_action_offsets_ptr, row_count) };
    let legal_ids = unsafe { slice::from_raw_parts(legal_action_ids_ptr, legal_action_ids_len) };
    let advantages_out = unsafe { slice::from_raw_parts_mut(advantages_ptr, row_count) };
    let returns_out = unsafe { slice::from_raw_parts_mut(returns_ptr, row_count) };
    let selected_local_out = unsafe { slice::from_raw_parts_mut(selected_local_ptr, row_count) };

    match prepare_ppo_batch(
        rewards,
        values,
        terminated,
        truncated,
        bootstrap_values,
        actions,
        counts,
        offsets,
        legal_ids,
        steps,
        env_count,
        gamma,
        gae_lambda,
        normalize,
        advantages_out,
        returns_out,
        selected_local_out,
    ) {
        Ok(()) => 0,
        Err(_) => -5,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_pad_legal_actions(
    legal_action_counts_ptr: *const usize,
    legal_action_offsets_ptr: *const usize,
    legal_action_ids_ptr: *const usize,
    legal_action_ids_len: usize,
    legal_action_features_ptr: *const f32,
    legal_action_features_len: usize,
    row_count: usize,
    feature_dim: usize,
    max_legal: usize,
    padded_ids_ptr: *mut usize,
    padded_features_ptr: *mut f32,
    legal_mask_ptr: *mut u8,
) -> i32 {
    if row_count == 0 || feature_dim == 0 || max_legal == 0 {
        return -1;
    }
    let Some(padded_rows) = row_count.checked_mul(max_legal) else {
        return -2;
    };
    let Some(padded_feature_len) = padded_rows.checked_mul(feature_dim) else {
        return -3;
    };
    if legal_action_counts_ptr.is_null()
        || legal_action_offsets_ptr.is_null()
        || legal_action_ids_ptr.is_null()
        || legal_action_features_ptr.is_null()
        || padded_ids_ptr.is_null()
        || padded_features_ptr.is_null()
        || legal_mask_ptr.is_null()
    {
        return -4;
    }

    let counts = unsafe { slice::from_raw_parts(legal_action_counts_ptr, row_count) };
    let offsets = unsafe { slice::from_raw_parts(legal_action_offsets_ptr, row_count) };
    let ids = unsafe { slice::from_raw_parts(legal_action_ids_ptr, legal_action_ids_len) };
    let features =
        unsafe { slice::from_raw_parts(legal_action_features_ptr, legal_action_features_len) };
    let padded_ids = unsafe { slice::from_raw_parts_mut(padded_ids_ptr, padded_rows) };
    let padded_features =
        unsafe { slice::from_raw_parts_mut(padded_features_ptr, padded_feature_len) };
    let legal_mask = unsafe { slice::from_raw_parts_mut(legal_mask_ptr, padded_rows) };

    match pad_legal_actions(
        counts,
        offsets,
        ids,
        features,
        row_count,
        feature_dim,
        max_legal,
        padded_ids,
        padded_features,
        legal_mask,
    ) {
        Ok(()) => 0,
        Err(_) => -5,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_pack_legal_action_rows(
    row_indices_ptr: *const usize,
    row_indices_len: usize,
    legal_action_counts_ptr: *const usize,
    legal_action_counts_len: usize,
    legal_action_offsets_ptr: *const usize,
    legal_action_offsets_len: usize,
    legal_action_ids_ptr: *const usize,
    legal_action_ids_len: usize,
    legal_action_features_ptr: *const f32,
    legal_action_features_len: usize,
    feature_dim: usize,
    packed_counts_ptr: *mut usize,
    packed_offsets_ptr: *mut usize,
    packed_ids_ptr: *mut usize,
    packed_ids_len: usize,
    packed_features_ptr: *mut f32,
    packed_features_len: usize,
) -> i32 {
    if row_indices_len == 0 || feature_dim == 0 {
        return -1;
    }
    if row_indices_ptr.is_null()
        || legal_action_counts_ptr.is_null()
        || legal_action_offsets_ptr.is_null()
        || legal_action_ids_ptr.is_null()
        || legal_action_features_ptr.is_null()
        || packed_counts_ptr.is_null()
        || packed_offsets_ptr.is_null()
        || packed_ids_ptr.is_null()
        || packed_features_ptr.is_null()
    {
        return -2;
    }

    let row_indices = unsafe { slice::from_raw_parts(row_indices_ptr, row_indices_len) };
    let counts = unsafe { slice::from_raw_parts(legal_action_counts_ptr, legal_action_counts_len) };
    let offsets =
        unsafe { slice::from_raw_parts(legal_action_offsets_ptr, legal_action_offsets_len) };
    let ids = unsafe { slice::from_raw_parts(legal_action_ids_ptr, legal_action_ids_len) };
    let features =
        unsafe { slice::from_raw_parts(legal_action_features_ptr, legal_action_features_len) };
    let packed_counts = unsafe { slice::from_raw_parts_mut(packed_counts_ptr, row_indices_len) };
    let packed_offsets = unsafe { slice::from_raw_parts_mut(packed_offsets_ptr, row_indices_len) };
    let packed_ids = unsafe { slice::from_raw_parts_mut(packed_ids_ptr, packed_ids_len) };
    let packed_features =
        unsafe { slice::from_raw_parts_mut(packed_features_ptr, packed_features_len) };

    match pack_legal_action_rows(
        row_indices,
        counts,
        offsets,
        ids,
        features,
        feature_dim,
        packed_counts,
        packed_offsets,
        packed_ids,
        packed_features,
    ) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_padded_argmax_actions(
    padded_logits_ptr: *const f32,
    padded_logits_len: usize,
    legal_action_counts_ptr: *const usize,
    legal_action_ids_ptr: *const usize,
    legal_action_ids_len: usize,
    row_count: usize,
    max_legal: usize,
    actions_ptr: *mut usize,
    selected_local_ptr: *mut i32,
    log_probs_ptr: *mut f32,
) -> i32 {
    if row_count == 0 || max_legal == 0 {
        return -1;
    }
    if padded_logits_ptr.is_null()
        || legal_action_counts_ptr.is_null()
        || legal_action_ids_ptr.is_null()
        || actions_ptr.is_null()
        || selected_local_ptr.is_null()
        || log_probs_ptr.is_null()
    {
        return -2;
    }

    let logits = unsafe { slice::from_raw_parts(padded_logits_ptr, padded_logits_len) };
    let counts = unsafe { slice::from_raw_parts(legal_action_counts_ptr, row_count) };
    let ids = unsafe { slice::from_raw_parts(legal_action_ids_ptr, legal_action_ids_len) };
    let actions = unsafe { slice::from_raw_parts_mut(actions_ptr, row_count) };
    let selected_local = unsafe { slice::from_raw_parts_mut(selected_local_ptr, row_count) };
    let log_probs = unsafe { slice::from_raw_parts_mut(log_probs_ptr, row_count) };

    match select_padded_argmax_actions(
        logits,
        counts,
        ids,
        row_count,
        max_legal,
        actions,
        selected_local,
        log_probs,
    ) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_compact_argmax_actions(
    legal_logits_ptr: *const f32,
    legal_logits_len: usize,
    legal_action_counts_ptr: *const usize,
    row_count: usize,
    legal_action_ids_ptr: *const usize,
    legal_action_ids_len: usize,
    actions_ptr: *mut usize,
    selected_local_ptr: *mut i32,
    log_probs_ptr: *mut f32,
) -> i32 {
    if row_count == 0 {
        return -1;
    }
    if legal_logits_ptr.is_null()
        || legal_action_counts_ptr.is_null()
        || legal_action_ids_ptr.is_null()
        || actions_ptr.is_null()
        || selected_local_ptr.is_null()
        || log_probs_ptr.is_null()
    {
        return -2;
    }

    let logits = unsafe { slice::from_raw_parts(legal_logits_ptr, legal_logits_len) };
    let counts = unsafe { slice::from_raw_parts(legal_action_counts_ptr, row_count) };
    let ids = unsafe { slice::from_raw_parts(legal_action_ids_ptr, legal_action_ids_len) };
    let actions = unsafe { slice::from_raw_parts_mut(actions_ptr, row_count) };
    let selected_local = unsafe { slice::from_raw_parts_mut(selected_local_ptr, row_count) };
    let log_probs = unsafe { slice::from_raw_parts_mut(log_probs_ptr, row_count) };

    match select_compact_argmax_actions(logits, counts, ids, actions, selected_local, log_probs) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_dense_argmax_actions(
    logits_ptr: *const f32,
    logits_len: usize,
    action_mask_ptr: *const u8,
    action_mask_len: usize,
    row_count: usize,
    action_count: usize,
    actions_ptr: *mut usize,
    log_probs_ptr: *mut f32,
) -> i32 {
    if row_count == 0 || action_count == 0 {
        return -1;
    }
    if logits_ptr.is_null()
        || action_mask_ptr.is_null()
        || actions_ptr.is_null()
        || log_probs_ptr.is_null()
    {
        return -2;
    }

    let logits = unsafe { slice::from_raw_parts(logits_ptr, logits_len) };
    let mask = unsafe { slice::from_raw_parts(action_mask_ptr, action_mask_len) };
    let actions = unsafe { slice::from_raw_parts_mut(actions_ptr, row_count) };
    let log_probs = unsafe { slice::from_raw_parts_mut(log_probs_ptr, row_count) };

    match select_dense_argmax_actions(logits, mask, row_count, action_count, actions, log_probs) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_repeat_row_indices(
    legal_action_counts_ptr: *const usize,
    row_count: usize,
    row_indices_ptr: *mut i32,
    row_indices_len: usize,
) -> i32 {
    if row_count == 0 {
        return -1;
    }
    if legal_action_counts_ptr.is_null() || row_indices_ptr.is_null() {
        return -2;
    }

    let counts = unsafe { slice::from_raw_parts(legal_action_counts_ptr, row_count) };
    let row_indices = unsafe { slice::from_raw_parts_mut(row_indices_ptr, row_indices_len) };

    match repeat_row_indices(counts, row_indices) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_normalize_legal_offsets(
    legal_action_counts_ptr: *const usize,
    legal_action_offsets_ptr: *const usize,
    row_count: usize,
    normalized_offsets_ptr: *mut usize,
) -> i32 {
    if row_count == 0 {
        return -1;
    }
    if legal_action_counts_ptr.is_null()
        || legal_action_offsets_ptr.is_null()
        || normalized_offsets_ptr.is_null()
    {
        return -2;
    }

    let counts = unsafe { slice::from_raw_parts(legal_action_counts_ptr, row_count) };
    let offsets = unsafe { slice::from_raw_parts(legal_action_offsets_ptr, row_count) };
    let normalized = unsafe { slice::from_raw_parts_mut(normalized_offsets_ptr, row_count) };

    match normalize_legal_offsets(counts, offsets, normalized) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

impl FfiWorker {
    fn new(worker: BatchedRolloutWorker, action_features_dtype: TensorDType) -> Self {
        let last = worker.encode_all();
        let action_features_f16 = if action_features_dtype == TensorDType::Float16 {
            f32_to_f16_bits(&last.action_features)
        } else {
            Vec::new()
        };
        let legal_action_features_f16 = if action_features_dtype == TensorDType::Float16 {
            f32_to_f16_bits(&last.legal_action_features)
        } else {
            Vec::new()
        };
        let terminated_u8 = terminated_to_u8(&last.terminated);
        let truncated_u8 = terminated_to_u8(&last.truncated);
        let reset_flags_u8 = terminated_to_u8(&last.reset_flags);
        let terminal_observation_valid_u8 = terminated_to_u8(&last.terminal_observation_valid);
        let mana_draw_legal_u8 = terminated_to_u8(&last.mana_draw_legal);
        Self {
            worker,
            last,
            action_features_dtype,
            action_features_f16,
            legal_action_features_f16,
            terminated_u8,
            truncated_u8,
            reset_flags_u8,
            terminal_observation_valid_u8,
            mana_draw_legal_u8,
        }
    }

    fn set_last(&mut self, output: BatchTensorOutput) {
        if self.action_features_dtype == TensorDType::Float16 {
            self.action_features_f16 = f32_to_f16_bits(&output.action_features);
            self.legal_action_features_f16 = f32_to_f16_bits(&output.legal_action_features);
        }
        self.terminated_u8 = terminated_to_u8(&output.terminated);
        self.truncated_u8 = terminated_to_u8(&output.truncated);
        self.reset_flags_u8 = terminated_to_u8(&output.reset_flags);
        self.terminal_observation_valid_u8 = terminated_to_u8(&output.terminal_observation_valid);
        self.mana_draw_legal_u8 = terminated_to_u8(&output.mana_draw_legal);
        self.last = output;
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
) -> *mut FfiWorker {
    unsafe { trainv3_worker_from_trace_json_with_options(json_ptr, json_len, env_count, 0) }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_with_options(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
) -> *mut FfiWorker {
    unsafe {
        trainv3_worker_from_trace_json_with_options_v2(
            json_ptr,
            json_len,
            env_count,
            action_features_dtype_code,
            0,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_with_options_v2(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
) -> *mut FfiWorker {
    unsafe {
        trainv3_worker_from_trace_json_with_options_v3(
            json_ptr,
            json_len,
            env_count,
            action_features_dtype_code,
            action_feature_output_code,
            0,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_with_options_v3(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
    observation_output_code: u32,
) -> *mut FfiWorker {
    unsafe {
        trainv3_worker_from_trace_json_with_options_v4(
            json_ptr,
            json_len,
            env_count,
            action_features_dtype_code,
            action_feature_output_code,
            observation_output_code,
            0,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_with_options_v4(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
    observation_output_code: u32,
    action_mask_output_code: u32,
) -> *mut FfiWorker {
    unsafe {
        trainv3_worker_from_trace_json_with_options_v5(
            json_ptr,
            json_len,
            env_count,
            action_features_dtype_code,
            action_feature_output_code,
            observation_output_code,
            action_mask_output_code,
            0,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_with_options_v5(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
    observation_output_code: u32,
    action_mask_output_code: u32,
    terminal_observation_output_code: u32,
) -> *mut FfiWorker {
    unsafe {
        trainv3_worker_from_trace_json_with_options_v6(
            json_ptr,
            json_len,
            env_count,
            action_features_dtype_code,
            action_feature_output_code,
            observation_output_code,
            action_mask_output_code,
            terminal_observation_output_code,
            0,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_with_options_v6(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
    observation_output_code: u32,
    action_mask_output_code: u32,
    terminal_observation_output_code: u32,
    diagnostic_output_code: u32,
) -> *mut FfiWorker {
    if json_ptr.is_null() || json_len == 0 || env_count == 0 {
        return ptr::null_mut();
    }
    let Some(action_features_dtype) = tensor_dtype_from_code(action_features_dtype_code) else {
        return ptr::null_mut();
    };
    let Some(action_feature_output) = action_feature_output_from_code(action_feature_output_code)
    else {
        return ptr::null_mut();
    };
    let Some(observation_output) = observation_output_from_code(observation_output_code) else {
        return ptr::null_mut();
    };
    let Some(action_mask_output) = action_mask_output_from_code(action_mask_output_code) else {
        return ptr::null_mut();
    };
    let Some(terminal_observation_output) =
        terminal_observation_output_from_code(terminal_observation_output_code)
    else {
        return ptr::null_mut();
    };
    let Some(diagnostic_output) = diagnostic_output_from_code(diagnostic_output_code) else {
        return ptr::null_mut();
    };

    let raw = unsafe { slice::from_raw_parts(json_ptr, json_len) };
    let Ok(trace) = serde_json::from_slice::<GoldenTrace>(raw) else {
        return ptr::null_mut();
    };
    let config = KernelConfig::from_trace_config(&trace.env_config);
    let snapshots = vec![trace.initial; env_count];
    let worker = BatchedRolloutWorker::from_snapshots_with_outputs(
        config,
        &snapshots,
        action_feature_output,
        observation_output,
        action_mask_output,
        terminal_observation_output,
        diagnostic_output,
    );
    Box::into_raw(Box::new(FfiWorker::new(worker, action_features_dtype)))
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_pool_with_options_v2(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
) -> *mut FfiWorker {
    unsafe {
        trainv3_worker_from_trace_json_pool_with_options_v3(
            json_ptr,
            json_len,
            env_count,
            action_features_dtype_code,
            action_feature_output_code,
            0,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_pool_with_options_v3(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
    reset_pool_mode_code: u32,
) -> *mut FfiWorker {
    unsafe {
        trainv3_worker_from_trace_json_pool_with_options_v4(
            json_ptr,
            json_len,
            env_count,
            action_features_dtype_code,
            action_feature_output_code,
            reset_pool_mode_code,
            0,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_pool_with_options_v4(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
    reset_pool_mode_code: u32,
    observation_output_code: u32,
) -> *mut FfiWorker {
    unsafe {
        trainv3_worker_from_trace_json_pool_with_options_v5(
            json_ptr,
            json_len,
            env_count,
            action_features_dtype_code,
            action_feature_output_code,
            reset_pool_mode_code,
            observation_output_code,
            0,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_pool_with_options_v5(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
    reset_pool_mode_code: u32,
    observation_output_code: u32,
    action_mask_output_code: u32,
) -> *mut FfiWorker {
    unsafe {
        trainv3_worker_from_trace_json_pool_with_options_v6(
            json_ptr,
            json_len,
            env_count,
            action_features_dtype_code,
            action_feature_output_code,
            reset_pool_mode_code,
            observation_output_code,
            action_mask_output_code,
            0,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_pool_with_options_v6(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
    reset_pool_mode_code: u32,
    observation_output_code: u32,
    action_mask_output_code: u32,
    terminal_observation_output_code: u32,
) -> *mut FfiWorker {
    unsafe {
        trainv3_worker_from_trace_json_pool_with_options_v7(
            json_ptr,
            json_len,
            env_count,
            action_features_dtype_code,
            action_feature_output_code,
            reset_pool_mode_code,
            observation_output_code,
            action_mask_output_code,
            terminal_observation_output_code,
            0,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_from_trace_json_pool_with_options_v7(
    json_ptr: *const u8,
    json_len: usize,
    env_count: usize,
    action_features_dtype_code: u32,
    action_feature_output_code: u32,
    reset_pool_mode_code: u32,
    observation_output_code: u32,
    action_mask_output_code: u32,
    terminal_observation_output_code: u32,
    diagnostic_output_code: u32,
) -> *mut FfiWorker {
    if json_ptr.is_null() || json_len == 0 || env_count == 0 {
        return ptr::null_mut();
    }
    let Some(action_features_dtype) = tensor_dtype_from_code(action_features_dtype_code) else {
        return ptr::null_mut();
    };
    let Some(action_feature_output) = action_feature_output_from_code(action_feature_output_code)
    else {
        return ptr::null_mut();
    };
    let Some(cycle_resets) = reset_pool_cycle_from_code(reset_pool_mode_code) else {
        return ptr::null_mut();
    };
    let Some(observation_output) = observation_output_from_code(observation_output_code) else {
        return ptr::null_mut();
    };
    let Some(action_mask_output) = action_mask_output_from_code(action_mask_output_code) else {
        return ptr::null_mut();
    };
    let Some(terminal_observation_output) =
        terminal_observation_output_from_code(terminal_observation_output_code)
    else {
        return ptr::null_mut();
    };
    let Some(diagnostic_output) = diagnostic_output_from_code(diagnostic_output_code) else {
        return ptr::null_mut();
    };

    let raw = unsafe { slice::from_raw_parts(json_ptr, json_len) };
    let Ok(traces) = serde_json::from_slice::<Vec<GoldenTrace>>(raw) else {
        return ptr::null_mut();
    };
    if traces.is_empty() {
        return ptr::null_mut();
    }

    let config = KernelConfig::from_trace_config(&traces[0].env_config);
    let snapshots = traces
        .iter()
        .map(|trace| (trace.initial.clone(), trace.env_config.clone()))
        .collect::<Vec<_>>();
    let Ok(worker) = BatchedRolloutWorker::from_snapshot_pool_with_trace_configs(
        config,
        &snapshots,
        env_count,
        action_feature_output,
        observation_output,
        action_mask_output,
        terminal_observation_output,
        diagnostic_output,
        cycle_resets,
    ) else {
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(FfiWorker::new(worker, action_features_dtype)))
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_free(worker: *mut FfiWorker) {
    if worker.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(worker));
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_encode(worker: *mut FfiWorker) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    let output = worker.worker.encode_all();
    worker.set_last(output);
    0
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_reset(worker: *mut FfiWorker) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    worker.worker.reset_all();
    let output = worker.worker.encode_all();
    worker.set_last(output);
    0
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_use_chacha_rng(worker: *mut FfiWorker) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    worker.worker.use_chacha_rng();
    0
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_reset_indices(
    worker: *mut FfiWorker,
    indices_ptr: *const usize,
    indices_len: usize,
) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    if indices_len > 0 && indices_ptr.is_null() {
        return -2;
    }
    let indices = if indices_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(indices_ptr, indices_len) }
    };
    match worker.worker.reset_indices(indices) {
        Ok(()) => {
            let output = worker.worker.encode_all();
            worker.set_last(output);
            0
        }
        Err(_) => -3,
    }
}

/// Reset selected lanes without materialising a new full-batch observation.
///
/// A live PPO collector may reset several terminal lanes during one batched
/// step.  Encoding after every individual reset turns that into O(resets *
/// env_count) observation work; the caller can invoke `trainv3_worker_encode`
/// once after all resets instead.
#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_reset_indices_deferred(
    worker: *mut FfiWorker,
    indices_ptr: *const usize,
    indices_len: usize,
) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    if indices_len > 0 && indices_ptr.is_null() {
        return -2;
    }
    let indices = if indices_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(indices_ptr, indices_len) }
    };
    match worker.worker.reset_indices(indices) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_step(
    worker: *mut FfiWorker,
    actions_ptr: *const usize,
    actions_len: usize,
) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    if actions_ptr.is_null() {
        return -2;
    }
    let actions = unsafe { slice::from_raw_parts(actions_ptr, actions_len) };
    match worker.worker.step(actions) {
        Ok(output) => {
            worker.set_last(output);
            0
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_step_auto_reset(
    worker: *mut FfiWorker,
    actions_ptr: *const usize,
    actions_len: usize,
) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    if actions_ptr.is_null() {
        return -2;
    }
    let actions = unsafe { slice::from_raw_parts(actions_ptr, actions_len) };
    match worker.worker.step_auto_reset(actions) {
        Ok(output) => {
            worker.set_last(output);
            0
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_current_actor_ids(
    worker: *const FfiWorker,
    out_ptr: *mut i32,
    out_len: usize,
) -> i32 {
    let Some(worker) = worker_ref(worker) else {
        return -1;
    };
    if out_ptr.is_null() {
        return -2;
    }
    if out_len != worker.worker.env_count() {
        return -3;
    }
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, out_len) };
    for (slot, actor_id) in worker.worker.current_actor_ids().iter().copied().enumerate() {
        out[slot] = actor_id;
    }
    0
}

/// Per-env hero hp snapshot for the A4 live-self-play decisive-early-end
/// predicate (additive; mirrors `trainv3_worker_current_actor_ids`). Writes
/// `env_count * 4` i32 values laid out `[p1_hp, p1_max_hp, p2_hp, p2_max_hp]`
/// per env into `out_ptr`. Returns 0 on success, -1 null worker, -2 null out,
/// -3 length mismatch (`out_len != env_count * 4`).
#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_hero_hp(
    worker: *const FfiWorker,
    out_ptr: *mut i32,
    out_len: usize,
) -> i32 {
    let Some(worker) = worker_ref(worker) else {
        return -1;
    };
    if out_ptr.is_null() {
        return -2;
    }
    if out_len != worker.worker.env_count() * 4 {
        return -3;
    }
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, out_len) };
    let hero_hp = worker.worker.hero_hp();
    out.copy_from_slice(&hero_hp);
    0
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_select_rule_actions(
    worker: *const FfiWorker,
    agent_codes_ptr: *const u32,
    agent_codes_len: usize,
    salt: u64,
    out_ptr: *mut usize,
    out_len: usize,
) -> i32 {
    let Some(worker) = worker_ref(worker) else {
        return -1;
    };
    if agent_codes_ptr.is_null() || out_ptr.is_null() {
        return -2;
    }
    if agent_codes_len != worker.worker.env_count() || out_len != worker.worker.env_count() {
        return -3;
    }
    let agent_codes = unsafe { slice::from_raw_parts(agent_codes_ptr, agent_codes_len) };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, out_len) };
    match worker.worker.select_rule_actions(agent_codes, salt) {
        Ok(actions) => {
            out.copy_from_slice(&actions);
            0
        }
        Err(_) => -4,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_advance_rule_until_actor(
    worker: *mut FfiWorker,
    learner_actor_ids_ptr: *const i32,
    learner_actor_ids_len: usize,
    agent_codes_ptr: *const u32,
    agent_codes_len: usize,
    max_actions_per_env: usize,
    salt: u64,
    auto_reset: u8,
    learner_rewards_ptr: *mut f32,
    learner_rewards_len: usize,
    terminated_ptr: *mut u8,
    terminated_len: usize,
    truncated_ptr: *mut u8,
    truncated_len: usize,
    reset_flags_ptr: *mut u8,
    reset_flags_len: usize,
    action_counts_ptr: *mut usize,
    action_counts_len: usize,
) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    if learner_actor_ids_ptr.is_null()
        || agent_codes_ptr.is_null()
        || learner_rewards_ptr.is_null()
        || terminated_ptr.is_null()
        || reset_flags_ptr.is_null()
        || action_counts_ptr.is_null()
    {
        return -2;
    }
    let env_count = worker.worker.env_count();
    if learner_actor_ids_len != env_count
        || agent_codes_len != env_count
        || learner_rewards_len != env_count
        || terminated_len != env_count
        || reset_flags_len != env_count
        || action_counts_len != env_count
    {
        return -3;
    }
    // `truncated_ptr` may be NULL to preserve the legacy call signature for
    // callers that have not yet been updated (it is then ignored, matching the
    // nullable-output convention used by `trainv3_worker_step`'s optional
    // tensors). When non-NULL it must be `env_count` long.
    if !truncated_ptr.is_null() && truncated_len != env_count {
        return -3;
    }
    let auto_reset = match auto_reset {
        0 => false,
        1 => true,
        _ => return -4,
    };
    let learner_actor_ids =
        unsafe { slice::from_raw_parts(learner_actor_ids_ptr, learner_actor_ids_len) };
    let agent_codes = unsafe { slice::from_raw_parts(agent_codes_ptr, agent_codes_len) };
    let learner_rewards =
        unsafe { slice::from_raw_parts_mut(learner_rewards_ptr, learner_rewards_len) };
    let terminated = unsafe { slice::from_raw_parts_mut(terminated_ptr, terminated_len) };
    let truncated = if truncated_ptr.is_null() {
        None
    } else {
        Some(unsafe { slice::from_raw_parts_mut(truncated_ptr, truncated_len) })
    };
    let reset_flags = unsafe { slice::from_raw_parts_mut(reset_flags_ptr, reset_flags_len) };
    let action_counts = unsafe { slice::from_raw_parts_mut(action_counts_ptr, action_counts_len) };

    match worker.worker.advance_rule_until_actor(
        learner_actor_ids,
        agent_codes,
        max_actions_per_env,
        salt,
        auto_reset,
    ) {
        Ok(output) => {
            learner_rewards.copy_from_slice(&output.learner_rewards);
            for (idx, value) in output.terminated.iter().copied().enumerate() {
                terminated[idx] = u8::from(value);
            }
            if let Some(truncated_out) = truncated {
                for (idx, value) in output.truncated.iter().copied().enumerate() {
                    truncated_out[idx] = u8::from(value);
                }
            }
            for (idx, value) in output.reset_flags.iter().copied().enumerate() {
                reset_flags[idx] = u8::from(value);
            }
            action_counts.copy_from_slice(&output.action_counts);
            // `encode_all` seeds `BatchTensorOutput.truncated` with `false` for
            // every env (worker.rs:684) because it has no step result to read,
            // so `set_last` would clobber `truncated_u8` to all-false. The
            // advance path's truncation lives in `RuleAdvanceOutput.truncated`
            // (collected from `step.truncated` in the loop, mirroring the
            // learner-step path at worker.rs:814). Capture it BEFORE
            // `encode_all`/`set_last` and restore it AFTER, so
            // `trainv3_worker_truncated_ptr` reports the advance result's
            // truncation rather than `encode_all`'s placeholder. This mirrors
            // the learner-step path (worker.rs:814 +
            // `trainv3_worker_truncated_ptr`).
            let advance_truncated_u8 = terminated_to_u8(&output.truncated);
            let encoded = worker.worker.encode_all();
            worker.set_last(encoded);
            worker.truncated_u8 = advance_truncated_u8;
            0
        }
        Err(_) => -5,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_rollout_action_tape(
    worker: *mut FfiWorker,
    actions_ptr: *const usize,
    actions_len: usize,
    steps: usize,
    auto_reset: u8,
) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    if actions_len > 0 && actions_ptr.is_null() {
        return -2;
    }
    let auto_reset = match auto_reset {
        0 => false,
        1 => true,
        _ => return -4,
    };
    let actions = if actions_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(actions_ptr, actions_len) }
    };
    match worker
        .worker
        .rollout_action_tape(actions, steps, auto_reset)
    {
        Ok(output) => {
            worker.set_last(output);
            0
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_rollout_broadcast_action_ids(
    worker: *mut FfiWorker,
    action_ids_ptr: *const usize,
    action_ids_len: usize,
    auto_reset: u8,
) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    if action_ids_len > 0 && action_ids_ptr.is_null() {
        return -2;
    }
    let auto_reset = match auto_reset {
        0 => false,
        1 => true,
        _ => return -4,
    };
    let action_ids = if action_ids_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(action_ids_ptr, action_ids_len) }
    };
    match worker
        .worker
        .rollout_broadcast_action_ids(action_ids, auto_reset)
    {
        Ok(output) => {
            worker.set_last(output);
            0
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_rollout_action_tape_pre_step(
    worker: *mut FfiWorker,
    actions_ptr: *const usize,
    actions_len: usize,
    steps: usize,
    auto_reset: u8,
) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    if actions_len > 0 && actions_ptr.is_null() {
        return -2;
    }
    let auto_reset = match auto_reset {
        0 => false,
        1 => true,
        _ => return -4,
    };
    let actions = if actions_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(actions_ptr, actions_len) }
    };
    match worker
        .worker
        .rollout_action_tape_pre_step(actions, steps, auto_reset)
    {
        Ok(output) => {
            worker.set_last(output);
            0
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_rollout_broadcast_action_ids_pre_step(
    worker: *mut FfiWorker,
    action_ids_ptr: *const usize,
    action_ids_len: usize,
    auto_reset: u8,
) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    if action_ids_len > 0 && action_ids_ptr.is_null() {
        return -2;
    }
    let auto_reset = match auto_reset {
        0 => false,
        1 => true,
        _ => return -4,
    };
    let action_ids = if action_ids_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(action_ids_ptr, action_ids_len) }
    };
    match worker
        .worker
        .rollout_broadcast_action_ids_pre_step(action_ids, auto_reset)
    {
        Ok(output) => {
            worker.set_last(output);
            0
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_env_count(worker: *const FfiWorker) -> usize {
    worker_ref(worker).map(|w| w.last.env_count).unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_observation_v1_ptr(worker: *const FfiWorker) -> *const f32 {
    float_ptr(worker, |w| &w.last.observation_v1)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_observation_v1_len(worker: *const FfiWorker) -> usize {
    float_len(worker, |w| &w.last.observation_v1)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_observation_v5_ptr(worker: *const FfiWorker) -> *const f32 {
    float_ptr(worker, |w| &w.last.observation_v5)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_observation_v5_len(worker: *const FfiWorker) -> usize {
    float_len(worker, |w| &w.last.observation_v5)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_terminal_observation_v1_ptr(
    worker: *const FfiWorker,
) -> *const f32 {
    float_ptr(worker, |w| &w.last.terminal_observation_v1)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_terminal_observation_v1_len(
    worker: *const FfiWorker,
) -> usize {
    float_len(worker, |w| &w.last.terminal_observation_v1)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_terminal_observation_v5_ptr(
    worker: *const FfiWorker,
) -> *const f32 {
    float_ptr(worker, |w| &w.last.terminal_observation_v5)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_terminal_observation_v5_len(
    worker: *const FfiWorker,
) -> usize {
    float_len(worker, |w| &w.last.terminal_observation_v5)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_action_mask_ptr(worker: *const FfiWorker) -> *const f32 {
    float_ptr(worker, |w| &w.last.action_mask)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_action_mask_len(worker: *const FfiWorker) -> usize {
    float_len(worker, |w| &w.last.action_mask)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_action_features_ptr(
    worker: *const FfiWorker,
) -> *const f32 {
    float_ptr(worker, |w| &w.last.action_features)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_action_features_len(worker: *const FfiWorker) -> usize {
    float_len(worker, |w| &w.last.action_features)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_action_features_f16_ptr(
    worker: *const FfiWorker,
) -> *const u16 {
    worker_ref(worker)
        .map(|w| w.action_features_f16.as_ptr())
        .unwrap_or(ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_action_features_f16_len(worker: *const FfiWorker) -> usize {
    worker_ref(worker)
        .map(|w| w.action_features_f16.len())
        .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_legal_action_counts_ptr(
    worker: *const FfiWorker,
) -> *const usize {
    usize_ptr(worker, |w| &w.last.legal_action_counts)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_legal_action_counts_len(worker: *const FfiWorker) -> usize {
    usize_len(worker, |w| &w.last.legal_action_counts)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_legal_action_offsets_ptr(
    worker: *const FfiWorker,
) -> *const usize {
    usize_ptr(worker, |w| &w.last.legal_action_offsets)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_legal_action_offsets_len(
    worker: *const FfiWorker,
) -> usize {
    usize_len(worker, |w| &w.last.legal_action_offsets)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_legal_action_ids_ptr(
    worker: *const FfiWorker,
) -> *const usize {
    usize_ptr(worker, |w| &w.last.legal_action_ids)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_legal_action_ids_len(worker: *const FfiWorker) -> usize {
    usize_len(worker, |w| &w.last.legal_action_ids)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_legal_action_features_ptr(
    worker: *const FfiWorker,
) -> *const f32 {
    float_ptr(worker, |w| &w.last.legal_action_features)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_legal_action_features_len(
    worker: *const FfiWorker,
) -> usize {
    float_len(worker, |w| &w.last.legal_action_features)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_selected_local_indices_ptr(
    worker: *const FfiWorker,
) -> *const i32 {
    worker_ref(worker)
        .map(|w| w.last.selected_local_indices.as_ptr())
        .unwrap_or(ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_selected_local_indices_len(
    worker: *const FfiWorker,
) -> usize {
    worker_ref(worker)
        .map(|w| w.last.selected_local_indices.len())
        .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_legal_action_features_f16_ptr(
    worker: *const FfiWorker,
) -> *const u16 {
    worker_ref(worker)
        .map(|w| w.legal_action_features_f16.as_ptr())
        .unwrap_or(ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_legal_action_features_f16_len(
    worker: *const FfiWorker,
) -> usize {
    worker_ref(worker)
        .map(|w| w.legal_action_features_f16.len())
        .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_rewards_ptr(worker: *const FfiWorker) -> *const f32 {
    float_ptr(worker, |w| &w.last.rewards)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_rewards_len(worker: *const FfiWorker) -> usize {
    float_len(worker, |w| &w.last.rewards)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_counterparty_rewards_ptr(
    worker: *const FfiWorker,
) -> *const f32 {
    float_ptr(worker, |w| &w.last.counterparty_rewards)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_counterparty_rewards_len(
    worker: *const FfiWorker,
) -> usize {
    float_len(worker, |w| &w.last.counterparty_rewards)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_episode_returns_ptr(
    worker: *const FfiWorker,
) -> *const f32 {
    float_ptr(worker, |w| &w.last.episode_returns)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_episode_returns_len(worker: *const FfiWorker) -> usize {
    float_len(worker, |w| &w.last.episode_returns)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_episode_lengths_ptr(
    worker: *const FfiWorker,
) -> *const usize {
    usize_ptr(worker, |w| &w.last.episode_lengths)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_episode_lengths_len(worker: *const FfiWorker) -> usize {
    usize_len(worker, |w| &w.last.episode_lengths)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_terminated_ptr(worker: *const FfiWorker) -> *const u8 {
    worker_ref(worker)
        .map(|w| w.terminated_u8.as_ptr())
        .unwrap_or(ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_terminated_len(worker: *const FfiWorker) -> usize {
    worker_ref(worker)
        .map(|w| w.terminated_u8.len())
        .unwrap_or(0)
}

/// Per-env truncation flag (WD-2): each byte is 0/1, 1 means the post-step
/// `turn_number > max_turns` (mirroring
/// `ai/train_v2/classic_rl_env.py::ClassicRLEnv.step`'s `truncated`).
/// Length == env_count. Independent of `terminated`; both may be 1 if the
/// game ended on the same step that crossed the turn limit.
#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_truncated_ptr(worker: *const FfiWorker) -> *const u8 {
    worker_ref(worker)
        .map(|w| w.truncated_u8.as_ptr())
        .unwrap_or(ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_truncated_len(worker: *const FfiWorker) -> usize {
    worker_ref(worker)
        .map(|w| w.truncated_u8.len())
        .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_reset_flags_ptr(worker: *const FfiWorker) -> *const u8 {
    worker_ref(worker)
        .map(|w| w.reset_flags_u8.as_ptr())
        .unwrap_or(ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_reset_flags_len(worker: *const FfiWorker) -> usize {
    worker_ref(worker)
        .map(|w| w.reset_flags_u8.len())
        .unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_terminal_observation_valid_ptr(
    worker: *const FfiWorker,
) -> *const u8 {
    worker_ref(worker)
        .map(|w| w.terminal_observation_valid_u8.as_ptr())
        .unwrap_or(ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_terminal_observation_valid_len(
    worker: *const FfiWorker,
) -> usize {
    worker_ref(worker)
        .map(|w| w.terminal_observation_valid_u8.len())
        .unwrap_or(0)
}

/// Parallel binary mana_draw head — per-env legality flag for the current
/// actor's post-step observation (Phase 2: MD-3, AC-FFI-1/MD-FFI). Each byte
/// is 0/1: 1 means mana_draw is a legal action this turn for that env's
/// current player. Length == env_count. NOT part of the 601 action_mask.
#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_mana_draw_legal_ptr(
    worker: *const FfiWorker,
) -> *const u8 {
    worker_ref(worker)
        .map(|w| w.mana_draw_legal_u8.as_ptr())
        .unwrap_or(ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_mana_draw_legal_len(worker: *const FfiWorker) -> usize {
    worker_ref(worker)
        .map(|w| w.mana_draw_legal_u8.len())
        .unwrap_or(0)
}

/// Step the batch with a parallel mana_draw flag per env (Phase 2: MD-3).
/// `mana_draw_flags_ptr` points to `flags_len` bytes (0/1); when
/// `mana_draw_flags_ptr[i]` is non-zero, env `i` applies a mana-draw for its
/// current actor instead of decoding `actions_ptr[i]`. Pass a null pointer
/// (or all zeros) to behave like `trainv3_worker_step`. Returns 0 on
/// success, negative on error.
#[no_mangle]
pub unsafe extern "C" fn trainv3_worker_step_mana_draw(
    worker: *mut FfiWorker,
    actions_ptr: *const usize,
    actions_len: usize,
    mana_draw_flags_ptr: *const u8,
    flags_len: usize,
) -> i32 {
    let Some(worker) = worker_mut(worker) else {
        return -1;
    };
    if actions_ptr.is_null() {
        return -2;
    }
    let actions = unsafe { slice::from_raw_parts(actions_ptr, actions_len) };
    let flags: Vec<bool> = if mana_draw_flags_ptr.is_null() {
        vec![false; actions_len]
    } else {
        if flags_len != actions_len {
            return -5;
        }
        unsafe { slice::from_raw_parts(mana_draw_flags_ptr, flags_len) }
            .iter()
            .map(|b| *b != 0)
            .collect()
    };
    match worker.worker.step_with_mana_draw(actions, &flags, false) {
        Ok(output) => {
            worker.set_last(output);
            0
        }
        Err(err) => {
            eprintln!("trainv3_worker_step_mana_draw failed: {err}");
            -3
        }
    }
}

fn worker_ref<'a>(worker: *const FfiWorker) -> Option<&'a FfiWorker> {
    if worker.is_null() {
        None
    } else {
        Some(unsafe { &*worker })
    }
}

fn worker_mut<'a>(worker: *mut FfiWorker) -> Option<&'a mut FfiWorker> {
    if worker.is_null() {
        None
    } else {
        Some(unsafe { &mut *worker })
    }
}

fn float_ptr(worker: *const FfiWorker, get: fn(&FfiWorker) -> &Vec<f32>) -> *const f32 {
    worker_ref(worker)
        .map(|w| get(w).as_ptr())
        .unwrap_or(ptr::null())
}

fn float_len(worker: *const FfiWorker, get: fn(&FfiWorker) -> &Vec<f32>) -> usize {
    worker_ref(worker).map(|w| get(w).len()).unwrap_or(0)
}

fn usize_ptr(worker: *const FfiWorker, get: fn(&FfiWorker) -> &Vec<usize>) -> *const usize {
    worker_ref(worker)
        .map(|w| get(w).as_ptr())
        .unwrap_or(ptr::null())
}

fn usize_len(worker: *const FfiWorker, get: fn(&FfiWorker) -> &Vec<usize>) -> usize {
    worker_ref(worker).map(|w| get(w).len()).unwrap_or(0)
}

fn terminated_to_u8(values: &[bool]) -> Vec<u8> {
    values.iter().map(|v| u8::from(*v)).collect()
}

fn tensor_dtype_from_code(code: u32) -> Option<TensorDType> {
    match code {
        0 => Some(TensorDType::Float32),
        1 => Some(TensorDType::Float16),
        _ => None,
    }
}

fn action_feature_output_from_code(code: u32) -> Option<ActionFeatureOutput> {
    match code {
        0 => Some(ActionFeatureOutput::DenseAndLegal),
        1 => Some(ActionFeatureOutput::LegalOnly),
        _ => None,
    }
}

fn observation_output_from_code(code: u32) -> Option<ObservationOutput> {
    match code {
        0 => Some(ObservationOutput::V1AndV5),
        1 => Some(ObservationOutput::V5Only),
        _ => None,
    }
}

fn action_mask_output_from_code(code: u32) -> Option<ActionMaskOutput> {
    match code {
        0 => Some(ActionMaskOutput::Dense),
        1 => Some(ActionMaskOutput::LegalOnly),
        _ => None,
    }
}

fn terminal_observation_output_from_code(code: u32) -> Option<TerminalObservationOutput> {
    match code {
        0 => Some(TerminalObservationOutput::Full),
        1 => Some(TerminalObservationOutput::None),
        _ => None,
    }
}

fn diagnostic_output_from_code(code: u32) -> Option<DiagnosticOutput> {
    match code {
        0 => Some(DiagnosticOutput::Full),
        1 => Some(DiagnosticOutput::None),
        _ => None,
    }
}

fn reset_pool_cycle_from_code(code: u32) -> Option<bool> {
    match code {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

fn f32_to_f16_bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|value| f16::from_f32(*value).to_bits())
        .collect()
}
