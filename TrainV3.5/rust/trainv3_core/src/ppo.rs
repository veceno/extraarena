pub fn compute_gae_returns(
    rewards: &[f32],
    values: &[f32],
    terminated: &[u8],
    truncated: Option<&[u8]>,
    bootstrap_values: Option<&[f32]>,
    steps: usize,
    env_count: usize,
    gamma: f32,
    gae_lambda: f32,
    normalize_advantages: bool,
    advantages_out: &mut [f32],
    returns_out: &mut [f32],
) -> Result<(), String> {
    if steps == 0 || env_count == 0 {
        return Err("steps and env_count must be positive".to_string());
    }
    let row_count = steps
        .checked_mul(env_count)
        .ok_or_else(|| "steps * env_count overflowed".to_string())?;
    if rewards.len() != row_count
        || values.len() != row_count
        || terminated.len() != row_count
        || advantages_out.len() != row_count
        || returns_out.len() != row_count
    {
        return Err("rollout tensors must have length steps * env_count".to_string());
    }
    if truncated.is_some_and(|values| values.len() != row_count) {
        return Err("truncated must have length steps * env_count".to_string());
    }
    if bootstrap_values.is_some_and(|values| values.len() != env_count) {
        return Err("bootstrap_values length must match env_count".to_string());
    }

    for env_idx in 0..env_count {
        let mut last_gae = 0.0_f32;
        for step_idx in (0..steps).rev() {
            let index = step_idx * env_count + env_idx;
            let done = terminated[index] != 0 || truncated.is_some_and(|values| values[index] != 0);
            let nonterminal = if done { 0.0 } else { 1.0 };
            let next_value = if step_idx + 1 < steps {
                values[(step_idx + 1) * env_count + env_idx]
            } else {
                bootstrap_values.map_or(0.0, |values| values[env_idx])
            };
            let delta = rewards[index] + gamma * next_value * nonterminal - values[index];
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae;
            advantages_out[index] = last_gae;
        }
    }

    for idx in 0..row_count {
        returns_out[idx] = advantages_out[idx] + values[idx];
    }

    if normalize_advantages {
        let mean = advantages_out
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>()
            / row_count as f64;
        let variance = advantages_out
            .iter()
            .map(|value| {
                let centered = *value as f64 - mean;
                centered * centered
            })
            .sum::<f64>()
            / row_count as f64;
        let std = variance.sqrt();
        for value in advantages_out.iter_mut() {
            let centered = *value as f64 - mean;
            *value = if std > 1.0e-8 {
                (centered / (std + 1.0e-8)) as f32
            } else {
                centered as f32
            };
        }
    }

    Ok(())
}

pub fn select_local_indices(
    actions: &[usize],
    legal_action_counts: &[usize],
    legal_action_offsets: &[usize],
    legal_action_ids: &[usize],
    selected_out: &mut [i32],
) -> Result<(), String> {
    let row_count = actions.len();
    if legal_action_counts.len() != row_count
        || legal_action_offsets.len() != row_count
        || selected_out.len() != row_count
    {
        return Err(
            "actions, legal_action_counts, legal_action_offsets, and selected_out lengths must match"
                .to_string(),
        );
    }

    for row_idx in 0..row_count {
        let count = legal_action_counts[row_idx];
        let offset = legal_action_offsets[row_idx];
        if count == 0 {
            return Err(format!("row {row_idx} has no legal actions"));
        }
        let end = offset
            .checked_add(count)
            .ok_or_else(|| format!("row {row_idx} legal action range overflowed"))?;
        if end > legal_action_ids.len() {
            return Err(format!(
                "row {row_idx} legal action range [{offset}, {end}) exceeds ids length {}",
                legal_action_ids.len()
            ));
        }
        let action_id = actions[row_idx];
        let mut selected: Option<i32> = None;
        for (local_idx, legal_id) in legal_action_ids[offset..end].iter().enumerate() {
            if *legal_id == action_id {
                if selected.is_some() {
                    return Err(format!(
                        "action {action_id} appears more than once in legal ids for row {row_idx}"
                    ));
                }
                selected = Some(local_idx as i32);
            }
        }
        let Some(local_idx) = selected else {
            return Err(format!(
                "action {action_id} is absent from legal ids for row {row_idx}"
            ));
        };
        selected_out[row_idx] = local_idx;
    }

    Ok(())
}

pub fn prepare_ppo_batch(
    rewards: &[f32],
    values: &[f32],
    terminated: &[u8],
    truncated: Option<&[u8]>,
    bootstrap_values: Option<&[f32]>,
    actions: &[usize],
    legal_action_counts: &[usize],
    legal_action_offsets: &[usize],
    legal_action_ids: &[usize],
    steps: usize,
    env_count: usize,
    gamma: f32,
    gae_lambda: f32,
    normalize_advantages: bool,
    advantages_out: &mut [f32],
    returns_out: &mut [f32],
    selected_local_out: &mut [i32],
) -> Result<(), String> {
    let row_count = steps
        .checked_mul(env_count)
        .ok_or_else(|| "steps * env_count overflowed".to_string())?;
    if actions.len() != row_count
        || legal_action_counts.len() != row_count
        || legal_action_offsets.len() != row_count
        || selected_local_out.len() != row_count
    {
        return Err(
            "actions, legal-action rows, and selected_local_out must have length steps * env_count"
                .to_string(),
        );
    }

    compute_gae_returns(
        rewards,
        values,
        terminated,
        truncated,
        bootstrap_values,
        steps,
        env_count,
        gamma,
        gae_lambda,
        normalize_advantages,
        advantages_out,
        returns_out,
    )?;
    select_local_indices(
        actions,
        legal_action_counts,
        legal_action_offsets,
        legal_action_ids,
        selected_local_out,
    )
}

pub fn pad_legal_actions(
    legal_action_counts: &[usize],
    legal_action_offsets: &[usize],
    legal_action_ids: &[usize],
    legal_action_features: &[f32],
    row_count: usize,
    feature_dim: usize,
    max_legal: usize,
    padded_ids_out: &mut [usize],
    padded_features_out: &mut [f32],
    legal_mask_out: &mut [u8],
) -> Result<(), String> {
    if row_count == 0 || feature_dim == 0 || max_legal == 0 {
        return Err("row_count, feature_dim, and max_legal must be positive".to_string());
    }
    let padded_rows = row_count
        .checked_mul(max_legal)
        .ok_or_else(|| "row_count * max_legal overflowed".to_string())?;
    let padded_feature_len = padded_rows
        .checked_mul(feature_dim)
        .ok_or_else(|| "padded feature length overflowed".to_string())?;
    if legal_action_counts.len() != row_count || legal_action_offsets.len() != row_count {
        return Err(
            "legal_action_counts and legal_action_offsets must match row_count".to_string(),
        );
    }
    if padded_ids_out.len() != padded_rows
        || legal_mask_out.len() != padded_rows
        || padded_features_out.len() != padded_feature_len
    {
        return Err("padded output buffers have unexpected lengths".to_string());
    }
    let expected_feature_len = legal_action_ids
        .len()
        .checked_mul(feature_dim)
        .ok_or_else(|| "legal feature length overflowed".to_string())?;
    if legal_action_features.len() != expected_feature_len {
        return Err("legal_action_features length must equal ids_len * feature_dim".to_string());
    }

    padded_ids_out.fill(0);
    padded_features_out.fill(0.0);
    legal_mask_out.fill(0);

    for row_idx in 0..row_count {
        let count = legal_action_counts[row_idx];
        let offset = legal_action_offsets[row_idx];
        if count == 0 {
            return Err(format!("row {row_idx} has no legal actions"));
        }
        if count > max_legal {
            return Err(format!(
                "row {row_idx} count {count} exceeds max_legal {max_legal}"
            ));
        }
        let end = offset
            .checked_add(count)
            .ok_or_else(|| format!("row {row_idx} legal action range overflowed"))?;
        if end > legal_action_ids.len() {
            return Err(format!(
                "row {row_idx} legal action range [{offset}, {end}) exceeds ids length {}",
                legal_action_ids.len()
            ));
        }
        for local_idx in 0..count {
            let src_row = offset + local_idx;
            let dst_row = row_idx * max_legal + local_idx;
            padded_ids_out[dst_row] = legal_action_ids[src_row];
            legal_mask_out[dst_row] = 1;
            let src_start = src_row * feature_dim;
            let dst_start = dst_row * feature_dim;
            padded_features_out[dst_start..dst_start + feature_dim]
                .copy_from_slice(&legal_action_features[src_start..src_start + feature_dim]);
        }
    }

    Ok(())
}

pub fn pack_legal_action_rows(
    row_indices: &[usize],
    legal_action_counts: &[usize],
    legal_action_offsets: &[usize],
    legal_action_ids: &[usize],
    legal_action_features: &[f32],
    feature_dim: usize,
    packed_counts_out: &mut [usize],
    packed_offsets_out: &mut [usize],
    packed_ids_out: &mut [usize],
    packed_features_out: &mut [f32],
) -> Result<(), String> {
    let row_count = row_indices.len();
    if row_count == 0 || feature_dim == 0 {
        return Err("row_indices and feature_dim must be non-empty".to_string());
    }
    if packed_counts_out.len() != row_count || packed_offsets_out.len() != row_count {
        return Err("packed count/offset output lengths must match row_indices".to_string());
    }
    let expected_feature_len = legal_action_ids
        .len()
        .checked_mul(feature_dim)
        .ok_or_else(|| "legal feature length overflowed".to_string())?;
    if legal_action_features.len() != expected_feature_len {
        return Err("legal_action_features length must equal ids_len * feature_dim".to_string());
    }

    let mut packed_total = 0_usize;
    for (out_idx, row_idx) in row_indices.iter().copied().enumerate() {
        if row_idx >= legal_action_counts.len() || row_idx >= legal_action_offsets.len() {
            return Err(format!("row index {row_idx} exceeds legal row count"));
        }
        let count = legal_action_counts[row_idx];
        if count == 0 {
            return Err(format!("row {row_idx} has no legal actions"));
        }
        packed_counts_out[out_idx] = count;
        packed_offsets_out[out_idx] = packed_total;
        packed_total = packed_total
            .checked_add(count)
            .ok_or_else(|| "packed legal action count overflowed".to_string())?;
    }
    if packed_ids_out.len() != packed_total {
        return Err(format!(
            "packed_ids_out length {} does not match selected legal count {packed_total}",
            packed_ids_out.len()
        ));
    }
    let expected_packed_features = packed_total
        .checked_mul(feature_dim)
        .ok_or_else(|| "packed legal feature length overflowed".to_string())?;
    if packed_features_out.len() != expected_packed_features {
        return Err(format!(
            "packed_features_out length {} does not match selected feature length {expected_packed_features}",
            packed_features_out.len()
        ));
    }

    let mut dst_row = 0_usize;
    for row_idx in row_indices.iter().copied() {
        let count = legal_action_counts[row_idx];
        let offset = legal_action_offsets[row_idx];
        let end = offset
            .checked_add(count)
            .ok_or_else(|| format!("row {row_idx} legal action range overflowed"))?;
        if end > legal_action_ids.len() {
            return Err(format!(
                "row {row_idx} legal action range [{offset}, {end}) exceeds ids length {}",
                legal_action_ids.len()
            ));
        }
        packed_ids_out[dst_row..dst_row + count].copy_from_slice(&legal_action_ids[offset..end]);
        let src_feature_start = offset
            .checked_mul(feature_dim)
            .ok_or_else(|| format!("row {row_idx} feature offset overflowed"))?;
        let feature_len = count
            .checked_mul(feature_dim)
            .ok_or_else(|| format!("row {row_idx} feature length overflowed"))?;
        let dst_feature_start = dst_row
            .checked_mul(feature_dim)
            .ok_or_else(|| "packed feature offset overflowed".to_string())?;
        packed_features_out[dst_feature_start..dst_feature_start + feature_len].copy_from_slice(
            &legal_action_features[src_feature_start..src_feature_start + feature_len],
        );
        dst_row += count;
    }

    Ok(())
}

pub fn select_padded_argmax_actions(
    padded_logits: &[f32],
    legal_action_counts: &[usize],
    legal_action_ids: &[usize],
    row_count: usize,
    max_legal: usize,
    actions_out: &mut [usize],
    selected_local_out: &mut [i32],
    log_probs_out: &mut [f32],
) -> Result<(), String> {
    if row_count == 0 || max_legal == 0 {
        return Err("row_count and max_legal must be positive".to_string());
    }
    let expected_logits = row_count
        .checked_mul(max_legal)
        .ok_or_else(|| "padded logits length overflowed".to_string())?;
    if padded_logits.len() != expected_logits {
        return Err("padded_logits length must equal row_count * max_legal".to_string());
    }
    if legal_action_counts.len() != row_count
        || actions_out.len() != row_count
        || selected_local_out.len() != row_count
        || log_probs_out.len() != row_count
    {
        return Err("counts and output lengths must match row_count".to_string());
    }

    let mut legal_offset = 0_usize;
    for row_idx in 0..row_count {
        let count = legal_action_counts[row_idx];
        if count == 0 {
            return Err(format!("row {row_idx} has no legal actions"));
        }
        if count > max_legal {
            return Err(format!(
                "row {row_idx} count {count} exceeds max_legal {max_legal}"
            ));
        }
        let legal_end = legal_offset
            .checked_add(count)
            .ok_or_else(|| format!("row {row_idx} legal id range overflowed"))?;
        if legal_end > legal_action_ids.len() {
            return Err(format!(
                "row {row_idx} legal id range [{legal_offset}, {legal_end}) exceeds ids length {}",
                legal_action_ids.len()
            ));
        }

        let row_start = row_idx * max_legal;
        let row_logits = &padded_logits[row_start..row_start + count];
        let mut best_local = 0_usize;
        let mut best_logit = row_logits[0];
        for (local_idx, logit) in row_logits.iter().copied().enumerate().skip(1) {
            if logit > best_logit {
                best_local = local_idx;
                best_logit = logit;
            }
        }

        let exp_sum = row_logits
            .iter()
            .map(|logit| (*logit - best_logit).exp())
            .sum::<f32>();
        actions_out[row_idx] = legal_action_ids[legal_offset + best_local];
        selected_local_out[row_idx] = best_local as i32;
        log_probs_out[row_idx] = -exp_sum.ln();
        legal_offset = legal_end;
    }

    if legal_offset != legal_action_ids.len() {
        return Err(format!(
            "legal_action_ids has {} rows but counts describe {legal_offset}",
            legal_action_ids.len()
        ));
    }

    Ok(())
}

pub fn select_compact_argmax_actions(
    legal_logits: &[f32],
    legal_action_counts: &[usize],
    legal_action_ids: &[usize],
    actions_out: &mut [usize],
    selected_local_out: &mut [i32],
    log_probs_out: &mut [f32],
) -> Result<(), String> {
    let row_count = legal_action_counts.len();
    if row_count == 0 {
        return Err("legal_action_counts must be non-empty".to_string());
    }
    if actions_out.len() != row_count
        || selected_local_out.len() != row_count
        || log_probs_out.len() != row_count
    {
        return Err("output lengths must match legal_action_counts".to_string());
    }
    if legal_logits.len() != legal_action_ids.len() {
        return Err("legal_logits and legal_action_ids lengths must match".to_string());
    }

    let mut legal_offset = 0_usize;
    for row_idx in 0..row_count {
        let count = legal_action_counts[row_idx];
        if count == 0 {
            return Err(format!("row {row_idx} has no legal actions"));
        }
        let legal_end = legal_offset
            .checked_add(count)
            .ok_or_else(|| format!("row {row_idx} legal range overflowed"))?;
        if legal_end > legal_logits.len() {
            return Err(format!(
                "row {row_idx} legal range [{legal_offset}, {legal_end}) exceeds logits length {}",
                legal_logits.len()
            ));
        }

        let row_logits = &legal_logits[legal_offset..legal_end];
        let mut best_local = 0_usize;
        let mut best_logit = row_logits[0];
        for (local_idx, logit) in row_logits.iter().copied().enumerate().skip(1) {
            if logit > best_logit {
                best_local = local_idx;
                best_logit = logit;
            }
        }
        let exp_sum = row_logits
            .iter()
            .map(|logit| (*logit - best_logit).exp())
            .sum::<f32>();
        actions_out[row_idx] = legal_action_ids[legal_offset + best_local];
        selected_local_out[row_idx] = best_local as i32;
        log_probs_out[row_idx] = -exp_sum.ln();
        legal_offset = legal_end;
    }

    if legal_offset != legal_logits.len() {
        return Err(format!(
            "counts describe {legal_offset} legal rows but logits contain {}",
            legal_logits.len()
        ));
    }

    Ok(())
}

pub fn select_dense_argmax_actions(
    logits: &[f32],
    action_mask: &[u8],
    row_count: usize,
    action_count: usize,
    actions_out: &mut [usize],
    log_probs_out: &mut [f32],
) -> Result<(), String> {
    if row_count == 0 || action_count == 0 {
        return Err("row_count and action_count must be positive".to_string());
    }
    let expected_len = row_count
        .checked_mul(action_count)
        .ok_or_else(|| "dense logits length overflowed".to_string())?;
    if logits.len() != expected_len || action_mask.len() != expected_len {
        return Err(
            "logits and action_mask lengths must equal row_count * action_count".to_string(),
        );
    }
    if actions_out.len() != row_count || log_probs_out.len() != row_count {
        return Err("output lengths must match row_count".to_string());
    }

    for row_idx in 0..row_count {
        let row_start = row_idx * action_count;
        let mut best_action: Option<usize> = None;
        let mut best_logit = f32::NEG_INFINITY;
        for action_idx in 0..action_count {
            if action_mask[row_start + action_idx] == 0 {
                continue;
            }
            let logit = logits[row_start + action_idx];
            if best_action.is_none() || logit > best_logit {
                best_action = Some(action_idx);
                best_logit = logit;
            }
        }
        let Some(action_idx) = best_action else {
            return Err(format!("row {row_idx} has no legal actions"));
        };
        let exp_sum = (0..action_count)
            .filter(|action| action_mask[row_start + *action] != 0)
            .map(|action| (logits[row_start + action] - best_logit).exp())
            .sum::<f32>();
        actions_out[row_idx] = action_idx;
        log_probs_out[row_idx] = -exp_sum.ln();
    }

    Ok(())
}

pub fn repeat_row_indices(
    legal_action_counts: &[usize],
    row_indices_out: &mut [i32],
) -> Result<(), String> {
    if legal_action_counts.is_empty() {
        return Err("legal_action_counts must be non-empty".to_string());
    }
    let expected_len = legal_action_counts
        .iter()
        .try_fold(0_usize, |total, count| total.checked_add(*count))
        .ok_or_else(|| "legal action count overflowed".to_string())?;
    if expected_len != row_indices_out.len() {
        return Err(format!(
            "row_indices_out length {} does not match counts sum {expected_len}",
            row_indices_out.len()
        ));
    }

    let mut offset = 0_usize;
    for (row_idx, count) in legal_action_counts.iter().copied().enumerate() {
        if count == 0 {
            return Err(format!("row {row_idx} has no legal actions"));
        }
        let row_id = i32::try_from(row_idx)
            .map_err(|_| format!("row index {row_idx} does not fit int32"))?;
        for dst in &mut row_indices_out[offset..offset + count] {
            *dst = row_id;
        }
        offset += count;
    }

    Ok(())
}

pub fn normalize_legal_offsets(
    legal_action_counts: &[usize],
    legal_action_offsets: &[usize],
    normalized_offsets_out: &mut [usize],
) -> Result<(), String> {
    let row_count = legal_action_counts.len();
    if row_count == 0 {
        return Err("legal_action_counts must be non-empty".to_string());
    }
    if legal_action_offsets.len() != row_count || normalized_offsets_out.len() != row_count {
        return Err("counts, offsets, and normalized output lengths must match".to_string());
    }

    let base_offset = legal_action_offsets[0];
    let mut expected_offset = base_offset;
    let mut normalized_offset = 0_usize;
    for row_idx in 0..row_count {
        let count = legal_action_counts[row_idx];
        if count == 0 {
            return Err(format!("row {row_idx} has no legal actions"));
        }
        let offset = legal_action_offsets[row_idx];
        if offset != expected_offset {
            return Err(format!(
                "row {row_idx} offset {offset} does not match expected contiguous offset {expected_offset}"
            ));
        }
        normalized_offsets_out[row_idx] = normalized_offset;
        expected_offset = expected_offset
            .checked_add(count)
            .ok_or_else(|| "legal offset overflowed".to_string())?;
        normalized_offset = normalized_offset
            .checked_add(count)
            .ok_or_else(|| "normalized legal offset overflowed".to_string())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        compute_gae_returns, normalize_legal_offsets, pack_legal_action_rows, pad_legal_actions,
        prepare_ppo_batch, repeat_row_indices, select_compact_argmax_actions,
        select_dense_argmax_actions, select_local_indices, select_padded_argmax_actions,
    };

    #[test]
    fn gae_returns_stop_at_terminal_and_truncated_steps() {
        let rewards = [
            0.25, -0.1, 0.0, 0.5, 0.3, -0.2, -0.4, 0.7, 0.15, 1.0, -0.5, 0.25,
        ];
        let values = [
            0.1, -0.2, 0.05, 0.2, 0.4, -0.1, -0.3, 0.25, 0.35, 0.6, -0.15, 0.2,
        ];
        let terminated = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0];
        let truncated = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
        let bootstrap = [0.4, -0.2, 0.15];
        let mut advantages = [0.0_f32; 12];
        let mut returns = [0.0_f32; 12];

        compute_gae_returns(
            &rewards,
            &values,
            &terminated,
            Some(&truncated),
            Some(&bootstrap),
            4,
            3,
            0.91,
            0.77,
            false,
            &mut advantages,
            &mut returns,
        )
        .unwrap();

        assert!((advantages[4] - -0.1).abs() < 1.0e-6);
        assert!((advantages[8] - -0.2).abs() < 1.0e-6);
        assert!((advantages[9] - 0.4).abs() < 1.0e-6);
        assert!((returns[4] - 0.3).abs() < 1.0e-6);
        assert!((returns[8] - 0.15).abs() < 1.0e-6);
        assert!((returns[9] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn selected_local_indices_map_actions_into_variable_legal_tape() {
        let actions = [11, 23, 30, 43, 51, 62];
        let counts = [2, 3, 1, 4, 2, 3];
        let offsets = [0, 2, 5, 6, 10, 12];
        let legal_ids = [10, 11, 21, 22, 23, 30, 40, 41, 42, 43, 51, 52, 60, 61, 62];
        let mut selected = [0_i32; 6];

        select_local_indices(&actions, &counts, &offsets, &legal_ids, &mut selected).unwrap();

        assert_eq!(selected, [1, 2, 0, 3, 0, 2]);
    }

    #[test]
    fn prepare_ppo_batch_fuses_gae_and_selected_local_indices() {
        let rewards = [0.2, -0.1, 0.0, 0.4, 0.3, -0.2];
        let values = [0.1, -0.2, 0.05, 0.2, 0.4, -0.1];
        let terminated = [0_u8; 6];
        let truncated = [0_u8; 6];
        let bootstrap = [0.25, -0.15, 0.05];
        let actions = [11, 23, 30, 43, 51, 62];
        let counts = [2, 3, 1, 4, 2, 3];
        let offsets = [0, 2, 5, 6, 10, 12];
        let legal_ids = [10, 11, 21, 22, 23, 30, 40, 41, 42, 43, 51, 52, 60, 61, 62];
        let mut fused_advantages = [0.0_f32; 6];
        let mut fused_returns = [0.0_f32; 6];
        let mut fused_selected = [-1_i32; 6];
        let mut expected_advantages = [0.0_f32; 6];
        let mut expected_returns = [0.0_f32; 6];
        let mut expected_selected = [-1_i32; 6];

        prepare_ppo_batch(
            &rewards,
            &values,
            &terminated,
            Some(&truncated),
            Some(&bootstrap),
            &actions,
            &counts,
            &offsets,
            &legal_ids,
            2,
            3,
            0.93,
            0.81,
            true,
            &mut fused_advantages,
            &mut fused_returns,
            &mut fused_selected,
        )
        .unwrap();
        compute_gae_returns(
            &rewards,
            &values,
            &terminated,
            Some(&truncated),
            Some(&bootstrap),
            2,
            3,
            0.93,
            0.81,
            true,
            &mut expected_advantages,
            &mut expected_returns,
        )
        .unwrap();
        select_local_indices(
            &actions,
            &counts,
            &offsets,
            &legal_ids,
            &mut expected_selected,
        )
        .unwrap();

        assert_eq!(fused_selected, expected_selected);
        for idx in 0..6 {
            assert!((fused_advantages[idx] - expected_advantages[idx]).abs() < 1.0e-6);
            assert!((fused_returns[idx] - expected_returns[idx]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn pad_legal_actions_expands_variable_legal_tape() {
        let counts = [2, 3, 1];
        let offsets = [0, 2, 5];
        let ids = [10, 11, 20, 21, 22, 30];
        let features = (0..24).map(|value| value as f32).collect::<Vec<_>>();
        let mut padded_ids = [99_usize; 9];
        let mut padded_features = [99.0_f32; 36];
        let mut mask = [99_u8; 9];

        pad_legal_actions(
            &counts,
            &offsets,
            &ids,
            &features,
            3,
            4,
            3,
            &mut padded_ids,
            &mut padded_features,
            &mut mask,
        )
        .unwrap();

        assert_eq!(padded_ids, [10, 11, 0, 20, 21, 22, 30, 0, 0]);
        assert_eq!(mask, [1, 1, 0, 1, 1, 1, 1, 0, 0]);
        assert_eq!(&padded_features[0..8], &features[0..8]);
        assert_eq!(&padded_features[8..12], &[0.0; 4]);
        assert_eq!(&padded_features[12..24], &features[8..20]);
        assert_eq!(&padded_features[24..28], &features[20..24]);
        assert_eq!(&padded_features[28..36], &[0.0; 8]);
    }

    #[test]
    fn pack_legal_action_rows_compacts_selected_minibatch_rows() {
        let counts = [2, 3, 1, 4, 2];
        let offsets = [0, 2, 5, 6, 10];
        let ids = [10, 11, 20, 21, 22, 30, 40, 41, 42, 43, 50, 51];
        let features = (0..48).map(|value| value as f32).collect::<Vec<_>>();
        let row_indices = [3, 0, 4];
        let mut packed_counts = [0_usize; 3];
        let mut packed_offsets = [99_usize; 3];
        let mut packed_ids = [0_usize; 8];
        let mut packed_features = [0.0_f32; 32];

        pack_legal_action_rows(
            &row_indices,
            &counts,
            &offsets,
            &ids,
            &features,
            4,
            &mut packed_counts,
            &mut packed_offsets,
            &mut packed_ids,
            &mut packed_features,
        )
        .unwrap();

        assert_eq!(packed_counts, [4, 2, 2]);
        assert_eq!(packed_offsets, [0, 4, 6]);
        assert_eq!(packed_ids, [40, 41, 42, 43, 10, 11, 50, 51]);
        assert_eq!(&packed_features[0..16], &features[24..40]);
        assert_eq!(&packed_features[16..24], &features[0..8]);
        assert_eq!(&packed_features[24..32], &features[40..48]);
    }

    #[test]
    fn padded_argmax_selects_actions_and_log_probs() {
        let counts = [2, 3, 1, 2];
        let ids = [10, 11, 20, 21, 22, 30, 40, 41];
        let logits = [
            0.2, 1.2, -1.0e9, -0.4, 0.7, 0.1, 2.0, -1.0e9, -1.0e9, 0.5, 0.5, -1.0e9,
        ];
        let mut actions = [0_usize; 4];
        let mut selected_local = [-1_i32; 4];
        let mut log_probs = [0.0_f32; 4];

        select_padded_argmax_actions(
            &logits,
            &counts,
            &ids,
            4,
            3,
            &mut actions,
            &mut selected_local,
            &mut log_probs,
        )
        .unwrap();

        assert_eq!(actions, [11, 21, 30, 40]);
        assert_eq!(selected_local, [1, 1, 0, 0]);
        assert!((log_probs[0] - -0.31326166).abs() < 1.0e-6);
        assert!((log_probs[1] - -0.63216645).abs() < 1.0e-6);
        assert!((log_probs[2] - 0.0).abs() < 1.0e-6);
        assert!((log_probs[3] - -0.69314718).abs() < 1.0e-6);
    }

    #[test]
    fn compact_argmax_selects_actions_and_log_probs() {
        let counts = [2, 3, 1, 2];
        let ids = [10, 11, 20, 21, 22, 30, 40, 41];
        let logits = [0.2, 1.2, -0.4, 0.7, 0.1, 2.0, 0.5, 0.5];
        let mut actions = [0_usize; 4];
        let mut selected_local = [-1_i32; 4];
        let mut log_probs = [0.0_f32; 4];

        select_compact_argmax_actions(
            &logits,
            &counts,
            &ids,
            &mut actions,
            &mut selected_local,
            &mut log_probs,
        )
        .unwrap();

        assert_eq!(actions, [11, 21, 30, 40]);
        assert_eq!(selected_local, [1, 1, 0, 0]);
        assert!((log_probs[0] - -0.31326166).abs() < 1.0e-6);
        assert!((log_probs[1] - -0.63216645).abs() < 1.0e-6);
        assert!((log_probs[2] - 0.0).abs() < 1.0e-6);
        assert!((log_probs[3] - -0.69314718).abs() < 1.0e-6);
    }

    #[test]
    fn dense_argmax_respects_mask_and_computes_legal_log_probs() {
        let logits = [
            0.2, 1.2, -0.5, 0.1, 0.4, -0.4, 0.7, 0.1, 1.5, -0.2, 2.0, 0.0, 1.0, -0.3, 0.5,
        ];
        let mask = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0];
        let mut actions = [0_usize; 3];
        let mut log_probs = [0.0_f32; 3];

        select_dense_argmax_actions(&logits, &mask, 3, 5, &mut actions, &mut log_probs).unwrap();

        assert_eq!(actions, [1, 1, 2]);
        assert!((log_probs[0] - -0.5973014).abs() < 1.0e-6);
        assert!((log_probs[1] - -0.43748796).abs() < 1.0e-6);
        assert!((log_probs[2] - -0.24100845).abs() < 1.0e-6);
    }

    #[test]
    fn repeat_row_indices_expands_counts() {
        let counts = [2, 3, 1, 4];
        let mut row_indices = [0_i32; 10];

        repeat_row_indices(&counts, &mut row_indices).unwrap();

        assert_eq!(row_indices, [0, 0, 1, 1, 1, 2, 3, 3, 3, 3]);
    }

    #[test]
    fn normalize_legal_offsets_shifts_contiguous_offsets_to_zero_base() {
        let counts = [2, 3, 1, 4];
        let offsets = [10, 12, 15, 16];
        let mut normalized = [99_usize; 4];

        normalize_legal_offsets(&counts, &offsets, &mut normalized).unwrap();

        assert_eq!(normalized, [0, 2, 5, 6]);

        let err = normalize_legal_offsets(&counts, &[10, 13, 15, 16], &mut normalized)
            .expect_err("non-contiguous offsets must fail");
        assert!(err.contains("expected contiguous offset"));
    }
}
