"""Policy helpers for compact legal-action tensors from the Rust TrainV3 worker."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .rust_ffi import (
    compute_rust_compact_argmax_actions,
    compute_rust_dense_argmax_actions,
    compute_rust_pad_legal_actions,
    compute_rust_padded_argmax_actions,
    compute_rust_pack_legal_action_rows,
    compute_rust_repeat_row_indices,
)


@dataclass(frozen=True)
class CompactLegalActionScores:
    legal_logits: Any
    values: Any


@dataclass(frozen=True)
class PaddedLegalActionScores:
    padded_logits: Any
    legal_mask: Any
    values: Any
    profile: dict[str, float] | None = None


@dataclass(frozen=True)
class PaddedLegalActionInputs:
    padded_features: Any
    legal_mask: Any


def score_compact_legal_actions(
    model: Any,
    observations: Any,
    legal_action_counts: Any,
    legal_action_features: Any,
    *,
    row_index_backend: str = "rust",
    library_path: Any | None = None,
) -> CompactLegalActionScores:
    """Score only legal action rows with an ActionConditionedPolicy-compatible model.

    This mirrors `ActionConditionedPolicy.__call__`, but replaces the dense
    `(B, 601, action_dim)` action path with a flat legal-row path. `mlx` is
    imported lazily so importing TrainV3 helpers does not require Metal access.
    """
    import mlx.core as mx

    obs = observations if isinstance(observations, mx.array) else mx.array(observations)
    counts = _legal_counts_uintp_view_or_copy(legal_action_counts)
    total_legal = int(counts.sum())
    if counts.ndim != 1:
        raise ValueError(f"legal_action_counts must be 1D, got shape {counts.shape}")
    if obs.shape[0] != counts.shape[0]:
        raise ValueError(f"observation batch {obs.shape[0]} does not match counts {counts.shape[0]}")
    if total_legal <= 0:
        raise ValueError("at least one legal action row is required")
    if row_index_backend not in {"python", "rust"}:
        raise ValueError("row_index_backend must be python or rust")

    features = (
        legal_action_features
        if isinstance(legal_action_features, mx.array)
        else mx.array(legal_action_features)
    )
    if features.shape[0] != total_legal:
        raise ValueError(f"legal feature rows {features.shape[0]} do not match counts sum {total_legal}")
    if features.shape[1] != model.action_feature_dim:
        raise ValueError(
            f"legal feature dim {features.shape[1]} does not match model dim {model.action_feature_dim}"
        )

    state_emb = _encode_model_state(model, obs)
    if row_index_backend == "python":
        env_indices_np = np.repeat(
            np.arange(counts.shape[0], dtype=np.int32),
            counts.astype(np.int64, copy=False),
        )
    else:
        env_indices_np = compute_rust_repeat_row_indices(counts, library_path=library_path)
    env_indices = mx.array(env_indices_np)
    state_rows = state_emb[env_indices]
    action_emb = model.action_encoder(features)
    joint = mx.concatenate([state_rows, action_emb], axis=-1)
    legal_logits = model.candidate_scorer(joint).squeeze(-1)
    values = model.value_head(state_emb).squeeze(-1)
    return CompactLegalActionScores(legal_logits=legal_logits, values=values)


def score_padded_legal_actions(
    model: Any,
    observations: Any,
    legal_action_counts: Any,
    legal_action_features: Any,
    *,
    legal_action_offsets: Any | None = None,
    legal_action_ids: Any | None = None,
    padding_backend: str = "python",
    mask_invalid_logits: bool = True,
    profile_policy: bool = False,
    library_path: Any | None = None,
) -> PaddedLegalActionScores:
    """Score legal action rows after padding each env to the batch max legal count."""
    profile: dict[str, float] = {}
    padding_start = time.perf_counter() if profile_policy else 0.0
    padded_inputs = pad_legal_action_inputs(
        model,
        observations,
        legal_action_counts,
        legal_action_features,
        legal_action_offsets=legal_action_offsets,
        legal_action_ids=legal_action_ids,
        padding_backend=padding_backend,
        library_path=library_path,
    )
    if profile_policy:
        profile["policy_padding_seconds"] = time.perf_counter() - padding_start
    scores = score_padded_legal_action_inputs(
        model,
        observations,
        padded_inputs,
        mask_invalid_logits=mask_invalid_logits,
        profile_policy=profile_policy,
    )
    if scores.profile is not None:
        profile.update(scores.profile)
    if profile_policy:
        return PaddedLegalActionScores(
            padded_logits=scores.padded_logits,
            legal_mask=scores.legal_mask,
            values=scores.values,
            profile=profile,
        )
    return scores


def pad_legal_action_inputs(
    model: Any,
    observations: Any,
    legal_action_counts: Any,
    legal_action_features: Any,
    *,
    legal_action_offsets: Any | None = None,
    legal_action_ids: Any | None = None,
    padding_backend: str = "python",
    library_path: Any | None = None,
) -> PaddedLegalActionInputs:
    """Build model-independent padded legal-action features and mask."""
    obs_np = np.asarray(observations)
    counts = _legal_counts_uintp_view_or_copy(legal_action_counts)
    if counts.ndim != 1:
        raise ValueError(f"legal_action_counts must be 1D, got shape {counts.shape}")
    if obs_np.shape[0] != counts.shape[0]:
        raise ValueError(f"observation batch {obs_np.shape[0]} does not match counts {counts.shape[0]}")

    max_legal = int(counts.max(initial=0))
    total_legal = int(counts.sum())
    if max_legal <= 0 or total_legal <= 0:
        raise ValueError("at least one legal action row is required")

    features_np = np.asarray(legal_action_features)
    if features_np.shape != (total_legal, model.action_feature_dim):
        raise ValueError(
            f"legal features shape {features_np.shape} does not match "
            f"({total_legal}, {model.action_feature_dim})"
        )

    if padding_backend == "python":
        padded_features_np = np.zeros(
            (counts.shape[0], max_legal, model.action_feature_dim),
            dtype=features_np.dtype,
        )
        legal_mask_np = np.zeros((counts.shape[0], max_legal), dtype=np.bool_)
        offset = 0
        for env_idx, count in enumerate(counts.tolist()):
            if count <= 0:
                raise ValueError(f"env {env_idx} has no legal actions")
            padded_features_np[env_idx, :count] = features_np[offset : offset + count]
            legal_mask_np[env_idx, :count] = True
            offset += int(count)
    elif padding_backend == "rust":
        offsets = _legal_offsets_from_counts(counts) if legal_action_offsets is None else legal_action_offsets
        ids = np.arange(total_legal, dtype=np.uintp) if legal_action_ids is None else legal_action_ids
        padded = compute_rust_pad_legal_actions(
            counts,
            offsets,
            ids,
            features_np,
            library_path=library_path,
        )
        padded_features_np = padded.features
        legal_mask_np = padded.mask
    else:
        raise ValueError("padding_backend must be python or rust")

    return PaddedLegalActionInputs(padded_features=padded_features_np, legal_mask=legal_mask_np)


def score_padded_legal_action_inputs(
    model: Any,
    observations: Any,
    padded_inputs: PaddedLegalActionInputs,
    *,
    mask_invalid_logits: bool = True,
    profile_policy: bool = False,
) -> PaddedLegalActionScores:
    """Score pre-padded legal-action features without rebuilding the padded layout."""
    import mlx.core as mx

    obs = observations if isinstance(observations, mx.array) else mx.array(observations)
    padded_features = (
        padded_inputs.padded_features
        if isinstance(padded_inputs.padded_features, mx.array)
        else mx.array(padded_inputs.padded_features)
    )
    legal_mask = (
        padded_inputs.legal_mask
        if isinstance(padded_inputs.legal_mask, mx.array)
        else mx.array(padded_inputs.legal_mask)
    )
    if len(padded_features.shape) != 3:
        raise ValueError(f"padded legal features must be 3D, got shape {padded_features.shape}")
    if len(legal_mask.shape) != 2:
        raise ValueError(f"padded legal mask must be 2D, got shape {legal_mask.shape}")
    batch_size = obs.shape[0]
    if padded_features.shape[0] != batch_size:
        raise ValueError(
            f"padded feature batch {padded_features.shape[0]} does not match observations {batch_size}"
        )
    if legal_mask.shape != padded_features.shape[:2]:
        raise ValueError(
            f"padded mask shape {legal_mask.shape} does not match feature rows {padded_features.shape[:2]}"
        )
    if padded_features.shape[2] != model.action_feature_dim:
        raise ValueError(
            f"padded feature dim {padded_features.shape[2]} does not match model dim {model.action_feature_dim}"
        )
    model_start = time.perf_counter() if profile_policy else 0.0
    max_legal = int(padded_features.shape[1])
    state_emb = _encode_model_state(model, obs)

    action_emb = mx.reshape(padded_features, (batch_size * max_legal, model.action_feature_dim))
    action_emb = model.action_encoder(action_emb)
    action_emb = mx.reshape(action_emb, (batch_size, max_legal, model.action_hidden_dim))

    padded_logits = _score_action_conditioned_linear_candidates(
        model,
        state_emb,
        action_emb,
        batch_size=batch_size,
        max_legal=max_legal,
    )
    if padded_logits is None:
        state_bc = mx.expand_dims(state_emb, axis=1)
        state_bc = mx.broadcast_to(state_bc, (batch_size, max_legal, model.hidden_dim))
        joint = mx.concatenate([state_bc, action_emb], axis=-1)
        joint = mx.reshape(joint, (batch_size * max_legal, model.hidden_dim + model.action_hidden_dim))
        raw_logits = model.candidate_scorer(joint)
        padded_logits = mx.reshape(raw_logits, (batch_size, max_legal))
    if mask_invalid_logits:
        padded_logits = mx.where(legal_mask, padded_logits, mx.array(-1e9, dtype=padded_logits.dtype))
    values = model.value_head(state_emb).squeeze(-1)
    profile = None
    if profile_policy:
        mx.eval(padded_logits, values)
        profile = {"policy_model_seconds": time.perf_counter() - model_start}
    return PaddedLegalActionScores(
        padded_logits=padded_logits,
        legal_mask=legal_mask,
        values=values,
        profile=profile,
    )


def _score_action_conditioned_linear_candidates(
    model: Any,
    state_emb: Any,
    action_emb: Any,
    *,
    batch_size: int,
    max_legal: int,
) -> Any | None:
    scorer = getattr(model, "candidate_scorer", None)
    weight = getattr(scorer, "weight", None)
    if weight is None or len(weight.shape) != 2:
        return None

    expected_input_dim = int(model.hidden_dim) + int(model.action_hidden_dim)
    if int(weight.shape[0]) != 1 or int(weight.shape[1]) != expected_input_dim:
        return None

    import mlx.core as mx

    hidden_dim = int(model.hidden_dim)
    state_weight = weight[:, :hidden_dim]
    action_weight = weight[:, hidden_dim:]
    state_logits = state_emb @ mx.transpose(state_weight)
    flat_action_emb = mx.reshape(action_emb, (batch_size * max_legal, int(model.action_hidden_dim)))
    action_logits = flat_action_emb @ mx.transpose(action_weight)
    logits = mx.reshape(action_logits, (batch_size, max_legal)) + state_logits
    bias = getattr(scorer, "bias", None)
    if bias is not None:
        logits = logits + bias
    return logits


def _encode_model_state(model: Any, observations: Any) -> Any:
    encoder = getattr(model, "encode_state", None)
    if encoder is not None:
        return encoder(observations)
    state_encoder = getattr(model, "state_encoder", None)
    if state_encoder is None:
        raise AttributeError("model must provide encode_state(...) or state_encoder")
    return state_encoder(observations)


def compact_argmax_actions(
    legal_logits: Any,
    legal_action_counts: Any,
    legal_action_ids: Any,
) -> np.ndarray:
    """Pick the highest-logit action id per env from flat legal logits."""
    logits = np.asarray(legal_logits)
    counts = np.asarray(legal_action_counts, dtype=np.int64)
    ids = np.asarray(legal_action_ids, dtype=np.uintp)
    if logits.ndim != 1:
        raise ValueError(f"legal_logits must be 1D, got shape {logits.shape}")
    if counts.ndim != 1:
        raise ValueError(f"legal_action_counts must be 1D, got shape {counts.shape}")
    if ids.ndim != 1:
        raise ValueError(f"legal_action_ids must be 1D, got shape {ids.shape}")
    if logits.shape[0] != ids.shape[0] or logits.shape[0] != int(counts.sum()):
        raise ValueError("legal logits, ids, and counts describe different row counts")

    actions = np.empty(counts.shape[0], dtype=np.uintp)
    offset = 0
    for env_idx, count in enumerate(counts.tolist()):
        if count <= 0:
            raise ValueError(f"env {env_idx} has no legal actions")
        local = logits[offset : offset + count]
        actions[env_idx] = ids[offset + int(np.argmax(local))]
        offset += int(count)
    return actions


def padded_argmax_actions(
    padded_logits: Any,
    legal_action_counts: Any,
    legal_action_ids: Any,
) -> np.ndarray:
    """Pick the highest-logit action id per env from padded legal logits."""
    logits = np.asarray(padded_logits)
    counts = np.asarray(legal_action_counts, dtype=np.int64)
    ids = np.asarray(legal_action_ids, dtype=np.uintp)
    if logits.ndim != 2:
        raise ValueError(f"padded_logits must be 2D, got shape {logits.shape}")
    if counts.ndim != 1:
        raise ValueError(f"legal_action_counts must be 1D, got shape {counts.shape}")
    if ids.ndim != 1:
        raise ValueError(f"legal_action_ids must be 1D, got shape {ids.shape}")
    if logits.shape[0] != counts.shape[0] or ids.shape[0] != int(counts.sum()):
        raise ValueError("padded logits, ids, and counts describe different row counts")

    actions = np.empty(counts.shape[0], dtype=np.uintp)
    offset = 0
    for env_idx, count in enumerate(counts.tolist()):
        if count <= 0:
            raise ValueError(f"env {env_idx} has no legal actions")
        local = logits[env_idx, :count]
        actions[env_idx] = ids[offset + int(np.argmax(local))]
        offset += int(count)
    return actions


def make_padded_legal_argmax_policy(
    model: Any,
    *,
    padding_backend: str = "rust",
    selection_backend: str = "rust",
    padding_mode: str = "single",
    bucket_max_padding_ratio: float = 1.35,
    bucket_min_rows: int = 2048,
    bucket_pack_backend: str = "rust",
    profile_policy: bool = False,
    library_path: Any | None = None,
):
    """Create a collector policy that argmaxes over padded legal-action logits.

    The returned callable matches `collect_rust_vec_rollout`'s policy protocol
    and returns `actions`, `values`, and `log_probs` for PPO-style batches.
    """
    if selection_backend not in {"python", "rust"}:
        raise ValueError("selection_backend must be python or rust")
    if padding_mode not in {"single", "bucketed"}:
        raise ValueError("padding_mode must be single or bucketed")
    if bucket_pack_backend not in {"python", "rust"}:
        raise ValueError("bucket_pack_backend must be python or rust")

    def policy(observations: Any, _action_mask: Any, action_features: Any) -> dict[str, np.ndarray]:
        if padding_mode == "bucketed":
            return _bucketed_padded_argmax_policy_call(
                model,
                observations,
                action_features,
                padding_backend=padding_backend,
                selection_backend=selection_backend,
                bucket_max_padding_ratio=bucket_max_padding_ratio,
                bucket_min_rows=bucket_min_rows,
                bucket_pack_backend=bucket_pack_backend,
                profile_policy=profile_policy,
                library_path=library_path,
            )
        scores = score_padded_legal_actions(
            model,
            observations,
            action_features.counts,
            action_features.features,
            legal_action_offsets=action_features.offsets,
            legal_action_ids=action_features.ids,
            padding_backend=padding_backend,
            mask_invalid_logits=False,
            profile_policy=profile_policy,
            library_path=library_path,
        )
        profile = dict(scores.profile or {}) if profile_policy else {}
        selection_start = time.perf_counter() if profile_policy else 0.0
        logits = np.asarray(scores.padded_logits)
        counts = _legal_counts_uintp_view_or_copy(action_features.counts)
        values = np.asarray(scores.values, dtype=np.float32)
        ids = np.asarray(action_features.ids, dtype=np.uintp)
        if selection_backend == "python":
            actions = padded_argmax_actions(
                logits,
                counts,
                ids,
            )
            log_probs = np.empty(counts.shape[0], dtype=np.float32)
            selected_local_indices = np.empty(counts.shape[0], dtype=np.int32)
            offset = 0
            for env_idx, count in enumerate(counts.tolist()):
                if count <= 0:
                    raise ValueError(f"env {env_idx} has no legal actions")
                legal_ids = ids[offset : offset + count]
                local_logits = logits[env_idx, :count]
                chosen_local = int(np.argmax(local_logits))
                selected_local_indices[env_idx] = chosen_local
                if legal_ids[chosen_local] != actions[env_idx]:
                    raise RuntimeError("padded argmax action/id mismatch")
                shifted = local_logits - np.max(local_logits)
                log_probs[env_idx] = shifted[chosen_local] - np.log(np.exp(shifted).sum())
                offset += int(count)
        else:
            selected = compute_rust_padded_argmax_actions(
                logits,
                counts,
                ids,
                library_path=library_path,
            )
            actions = selected.actions
            log_probs = selected.log_probs
            selected_local_indices = selected.selected_local_indices
        if profile_policy:
            profile["policy_selection_seconds"] = time.perf_counter() - selection_start

        out = {
            "actions": actions,
            "values": values,
            "log_probs": log_probs,
            "selected_local_indices": selected_local_indices,
        }
        if profile_policy:
            out["policy_profile"] = profile
        return out

    return policy


def _bucketed_padded_argmax_policy_call(
    model: Any,
    observations: Any,
    action_features: Any,
    *,
    padding_backend: str,
    selection_backend: str,
    bucket_max_padding_ratio: float,
    bucket_min_rows: int,
    bucket_pack_backend: str,
    profile_policy: bool,
    library_path: Any | None,
) -> dict[str, np.ndarray]:
    counts = _legal_counts_uintp_view_or_copy(action_features.counts)
    buckets = _bucket_legal_action_indices(
        counts,
        max_padding_ratio=float(bucket_max_padding_ratio),
        min_bucket_size=int(bucket_min_rows),
    )
    if len(buckets) <= 1:
        single = make_padded_legal_argmax_policy(
            model,
            padding_backend=padding_backend,
            selection_backend=selection_backend,
            padding_mode="single",
            profile_policy=profile_policy,
            library_path=library_path,
        )
        return single(observations, None, action_features)

    obs_np = np.asarray(observations)
    offsets = np.asarray(action_features.offsets, dtype=np.uintp)
    ids = np.asarray(action_features.ids, dtype=np.uintp)
    features = np.asarray(action_features.features)
    batch_size = int(counts.shape[0])
    actions = np.empty(batch_size, dtype=np.uintp)
    values = np.empty(batch_size, dtype=np.float32)
    log_probs = np.empty(batch_size, dtype=np.float32)
    selected_local_indices = np.empty(batch_size, dtype=np.int32)
    profile: dict[str, float] = {}
    padded_rows = 0
    legal_rows = int(counts.sum(dtype=np.uintp))

    for bucket_indices in buckets:
        bucket_indices = np.asarray(bucket_indices, dtype=np.int64)
        bucket_counts = counts[bucket_indices]
        bucket_ids, bucket_features, bucket_offsets = _pack_bucket_legal_rows(
            bucket_indices,
            counts,
            offsets,
            ids,
            features,
            backend=bucket_pack_backend,
            library_path=library_path,
        )
        bucket_obs = obs_np[bucket_indices]
        scores = score_padded_legal_actions(
            model,
            bucket_obs,
            bucket_counts,
            bucket_features,
            legal_action_offsets=bucket_offsets,
            legal_action_ids=bucket_ids,
            padding_backend=padding_backend,
            mask_invalid_logits=False,
            profile_policy=profile_policy,
            library_path=library_path,
        )
        logits = np.asarray(scores.padded_logits)
        padded_rows += int(logits.shape[0]) * int(logits.shape[1])
        if selection_backend == "python":
            bucket_actions = padded_argmax_actions(logits, bucket_counts, bucket_ids)
            bucket_log_probs = np.empty(bucket_counts.shape[0], dtype=np.float32)
            bucket_selected = np.empty(bucket_counts.shape[0], dtype=np.int32)
            offset = 0
            for env_idx, count in enumerate(bucket_counts.tolist()):
                local_logits = logits[env_idx, : int(count)]
                chosen_local = int(np.argmax(local_logits))
                bucket_selected[env_idx] = chosen_local
                shifted = local_logits - np.max(local_logits)
                bucket_log_probs[env_idx] = shifted[chosen_local] - np.log(np.exp(shifted).sum())
                offset += int(count)
        else:
            selected = compute_rust_padded_argmax_actions(
                logits,
                bucket_counts,
                bucket_ids,
                library_path=library_path,
            )
            bucket_actions = selected.actions
            bucket_log_probs = selected.log_probs
            bucket_selected = selected.selected_local_indices
        actions[bucket_indices] = bucket_actions
        values[bucket_indices] = np.asarray(scores.values, dtype=np.float32)
        log_probs[bucket_indices] = bucket_log_probs
        selected_local_indices[bucket_indices] = bucket_selected
        if profile_policy and scores.profile is not None:
            for key, value in scores.profile.items():
                profile[key] = profile.get(key, 0.0) + float(value)

    out = {
        "actions": actions,
        "values": values,
        "log_probs": log_probs,
        "selected_local_indices": selected_local_indices,
    }
    if profile_policy:
        profile["policy_bucket_count"] = float(len(buckets))
        profile["policy_bucket_legal_rows"] = float(legal_rows)
        profile["policy_bucket_padded_rows"] = float(padded_rows)
        profile["policy_bucket_padding_ratio"] = 0.0 if legal_rows <= 0 else float(padded_rows) / float(legal_rows)
        out["policy_profile"] = profile
    return out


def _pack_bucket_legal_rows(
    bucket_indices: np.ndarray,
    source_counts: np.ndarray,
    source_offsets: np.ndarray,
    ids: np.ndarray,
    features: np.ndarray,
    *,
    backend: str,
    library_path: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if backend == "rust":
        packed = compute_rust_pack_legal_action_rows(
            bucket_indices,
            source_counts,
            source_offsets,
            ids,
            features,
            library_path=library_path,
        )
        return packed.ids, packed.features, packed.offsets

    counts = source_counts[bucket_indices]
    offsets = source_offsets[bucket_indices]
    new_offsets = np.empty(counts.shape[0], dtype=np.uintp)
    id_chunks: list[np.ndarray] = []
    feature_chunks: list[np.ndarray] = []
    total = 0
    for row_idx, (count, offset) in enumerate(zip(counts.tolist(), offsets.tolist())):
        count_i = int(count)
        offset_i = int(offset)
        if count_i <= 0:
            raise ValueError(f"bucket row {row_idx} has no legal actions")
        new_offsets[row_idx] = total
        id_chunks.append(ids[offset_i : offset_i + count_i])
        feature_chunks.append(features[offset_i : offset_i + count_i])
        total += count_i
    packed_ids = np.concatenate(id_chunks) if id_chunks else np.empty((0,), dtype=ids.dtype)
    packed_features = (
        np.concatenate(feature_chunks, axis=0)
        if feature_chunks
        else np.empty((0, features.shape[1]), dtype=features.dtype)
    )
    return packed_ids, packed_features, new_offsets


def _bucket_legal_action_indices(
    counts: Any,
    *,
    max_padding_ratio: float,
    min_bucket_size: int,
) -> list[np.ndarray]:
    counts_np = _legal_counts_uintp_view_or_copy(counts)
    if counts_np.ndim != 1:
        raise ValueError(f"legal_action_counts must be 1D, got shape {counts_np.shape}")
    if counts_np.size <= 1:
        return [np.arange(counts_np.size, dtype=np.int64)]
    if np.any(counts_np == 0):
        raise ValueError("legal action counts must be positive")
    ratio_limit = max(1.0, float(max_padding_ratio))
    min_size = max(1, int(min_bucket_size))
    order = np.argsort(counts_np, kind="stable").astype(np.int64, copy=False)
    buckets: list[np.ndarray] = []
    start = 0
    legal_sum = 0
    max_count = 0
    for pos, idx in enumerate(order):
        count = int(counts_np[int(idx)])
        new_size = pos - start + 1
        new_sum = legal_sum + count
        new_max = max(max_count, count)
        new_ratio = (new_size * new_max) / max(new_sum, 1)
        if pos > start and new_size > min_size and new_ratio > ratio_limit:
            buckets.append(order[start:pos].copy())
            start = pos
            legal_sum = count
            max_count = count
        else:
            legal_sum = new_sum
            max_count = new_max
    buckets.append(order[start:].copy())
    return buckets


def make_compact_legal_argmax_policy(
    model: Any,
    *,
    row_index_backend: str = "rust",
    selection_backend: str = "rust",
    library_path: Any | None = None,
):
    """Create a collector policy that scores only compact legal action rows."""
    if row_index_backend not in {"python", "rust"}:
        raise ValueError("row_index_backend must be python or rust")
    if selection_backend not in {"python", "rust"}:
        raise ValueError("selection_backend must be python or rust")

    def policy(observations: Any, _action_mask: Any, action_features: Any) -> dict[str, np.ndarray]:
        scores = score_compact_legal_actions(
            model,
            observations,
            action_features.counts,
            action_features.features,
            row_index_backend=row_index_backend,
            library_path=library_path,
        )
        logits = np.asarray(scores.legal_logits)
        counts = _legal_counts_uintp_view_or_copy(action_features.counts)
        ids = np.asarray(action_features.ids, dtype=np.uintp)
        values = np.asarray(scores.values, dtype=np.float32)
        if selection_backend == "python":
            actions = compact_argmax_actions(logits, counts, ids)
            log_probs = np.empty(counts.shape[0], dtype=np.float32)
            selected_local_indices = np.empty(counts.shape[0], dtype=np.int32)
            offset = 0
            for env_idx, count in enumerate(counts.tolist()):
                if count <= 0:
                    raise ValueError(f"env {env_idx} has no legal actions")
                local_logits = logits[offset : offset + count]
                chosen_local = int(np.argmax(local_logits))
                selected_local_indices[env_idx] = chosen_local
                shifted = local_logits - np.max(local_logits)
                log_probs[env_idx] = shifted[chosen_local] - np.log(np.exp(shifted).sum())
                offset += int(count)
        else:
            selected = compute_rust_compact_argmax_actions(
                logits,
                counts,
                ids,
                library_path=library_path,
            )
            actions = selected.actions
            log_probs = selected.log_probs
            selected_local_indices = selected.selected_local_indices

        return {
            "actions": actions,
            "values": values,
            "log_probs": log_probs,
            "selected_local_indices": selected_local_indices,
        }

    return policy


def make_dense_argmax_policy(
    model: Any,
    *,
    selection_backend: str = "python",
    library_path: Any | None = None,
):
    """Create a collector policy that argmaxes over dense 601-candidate logits."""
    if selection_backend not in {"python", "rust"}:
        raise ValueError("selection_backend must be python or rust")

    def policy(observations: Any, action_mask: Any, action_features: Any) -> dict[str, np.ndarray]:
        import mlx.core as mx

        obs = observations if isinstance(observations, mx.array) else mx.array(observations)
        features = action_features if isinstance(action_features, mx.array) else mx.array(action_features)
        logits, values = model(obs, features)
        logits_np = np.asarray(logits)
        values_np = np.asarray(values, dtype=np.float32)
        if selection_backend == "python":
            mask_np = np.asarray(action_mask, dtype=np.float32)
            masked_logits = np.where(mask_np > 0.0, logits_np, -1.0e9)
            actions = np.argmax(masked_logits, axis=1).astype(np.uintp)
            log_probs = np.empty(actions.shape[0], dtype=np.float32)

            for env_idx, action_id in enumerate(actions.tolist()):
                legal_logits = masked_logits[env_idx]
                shifted = legal_logits - np.max(legal_logits)
                log_probs[env_idx] = shifted[action_id] - np.log(np.exp(shifted).sum())
        else:
            selected = compute_rust_dense_argmax_actions(
                logits_np,
                action_mask,
                library_path=library_path,
            )
            actions = selected.actions
            log_probs = selected.log_probs

        return {
            "actions": actions,
            "values": values_np,
            "log_probs": log_probs,
        }

    return policy


def _legal_offsets_from_counts(counts: np.ndarray) -> np.ndarray:
    flat_counts = np.asarray(counts, dtype=np.uintp).reshape(-1)
    flat_offsets = np.empty_like(flat_counts, dtype=np.uintp)
    if flat_counts.size:
        flat_offsets[0] = 0
        if flat_counts.size > 1:
            flat_offsets[1:] = np.cumsum(flat_counts[:-1], dtype=np.uintp)
    return flat_offsets.reshape(counts.shape)


def _legal_counts_uintp_view_or_copy(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"legal_action_counts must be 1D, got shape {raw.shape}")
    if not np.issubdtype(raw.dtype, np.unsignedinteger) and np.any(raw < 0):
        raise ValueError("legal_action_counts must be non-negative")
    return np.ascontiguousarray(raw, dtype=np.uintp)


__all__ = [
    "CompactLegalActionScores",
    "PaddedLegalActionInputs",
    "PaddedLegalActionScores",
    "compact_argmax_actions",
    "make_dense_argmax_policy",
    "make_compact_legal_argmax_policy",
    "make_padded_legal_argmax_policy",
    "pad_legal_action_inputs",
    "padded_argmax_actions",
    "score_compact_legal_actions",
    "score_padded_legal_action_inputs",
    "score_padded_legal_actions",
]
