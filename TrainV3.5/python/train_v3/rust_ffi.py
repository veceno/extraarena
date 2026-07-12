"""ctypes bridge to the training-only Rust TrainV3 batch worker."""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ACTION_FEATURE_DIM, MAX_CANDIDATE_ACTIONS, OBS_V1_DIM, OBS_V5_DIM


@dataclass(frozen=True)
class RustPaddedLegalActions:
    features: np.ndarray
    ids: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class RustPackedLegalRows:
    counts: np.ndarray
    offsets: np.ndarray
    ids: np.ndarray
    features: np.ndarray


@dataclass(frozen=True)
class RustPaddedArgmaxActions:
    actions: np.ndarray
    selected_local_indices: np.ndarray
    log_probs: np.ndarray


@dataclass(frozen=True)
class RustCompactArgmaxActions:
    actions: np.ndarray
    selected_local_indices: np.ndarray
    log_probs: np.ndarray


@dataclass(frozen=True)
class RustDenseArgmaxActions:
    actions: np.ndarray
    log_probs: np.ndarray


@dataclass(frozen=True)
class RustPreparedPPOBatch:
    advantages: np.ndarray
    returns: np.ndarray
    selected_local_indices: np.ndarray


def _bool_view_from_u8(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.uint8:
        raise RuntimeError(f"expected Rust flag buffer to be uint8, got {values.dtype}")
    if values.dtype.itemsize != np.dtype(np.bool_).itemsize:
        raise RuntimeError("Rust flag buffers cannot be viewed as numpy bool arrays")
    return values.view(np.bool_)


def _u8_view_from_bool_or_u8(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.dtype == np.bool_:
        if arr.dtype.itemsize != np.dtype(np.uint8).itemsize:
            raise RuntimeError(f"{name} bool tape cannot be viewed as a Rust u8 flag buffer")
        if arr.flags.c_contiguous:
            return arr.view(np.uint8)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _dense_mask_u8_view_or_copy(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.dtype == np.bool_:
        if arr.dtype.itemsize != np.dtype(np.uint8).itemsize:
            raise RuntimeError("dense bool action masks cannot be viewed as Rust u8 masks")
        if arr.flags.c_contiguous:
            return arr.view(np.uint8)
    return np.ascontiguousarray(arr > 0, dtype=np.uint8)


def default_library_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    release = root / "target" / "release"
    if sys.platform == "darwin":
        names = ["libtrainv3_core.dylib"]
    elif os.name == "nt":
        names = ["trainv3_core.dll"]
    else:
        names = ["libtrainv3_core.so"]
    return [release / name for name in names]


def resolve_library_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        cache_key = ("explicit", os.fspath(path))
        cached = _RESOLVED_LIBRARY_PATH_CACHE.get(cache_key)
        if cached is not None:
            return cached
        candidate = Path(path)
        if candidate.exists():
            _RESOLVED_LIBRARY_PATH_CACHE[cache_key] = candidate
            return candidate
        raise FileNotFoundError(candidate)

    env_path = os.environ.get("TRAINV3_CORE_LIB")
    if env_path:
        cache_key = ("env", env_path)
        cached = _RESOLVED_LIBRARY_PATH_CACHE.get(cache_key)
        if cached is not None:
            return cached
        candidate = Path(env_path)
        if candidate.exists():
            _RESOLVED_LIBRARY_PATH_CACHE[cache_key] = candidate
            return candidate
        raise FileNotFoundError(candidate)

    cache_key = ("default", "")
    cached = _RESOLVED_LIBRARY_PATH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    for candidate in default_library_candidates():
        if candidate.exists():
            _RESOLVED_LIBRARY_PATH_CACHE[cache_key] = candidate
            return candidate

    searched = ", ".join(str(p) for p in default_library_candidates())
    raise FileNotFoundError(f"trainv3_core dynamic library not found; searched: {searched}")


_RESOLVED_LIBRARY_PATH_CACHE: dict[tuple[str, str], Path] = {}
_LIBRARY_CACHE: dict[Path, ctypes.CDLL] = {}


def compute_rust_gae_returns(
    rewards,
    values,
    terminated,
    truncated=None,
    *,
    bootstrap_values=None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
    library_path: str | os.PathLike[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PPO GAE advantages and returns through the Rust FFI kernel."""
    rewards_arr = np.ascontiguousarray(rewards, dtype=np.float32)
    values_arr = np.ascontiguousarray(values, dtype=np.float32)
    terminated_arr = _u8_view_from_bool_or_u8(terminated, "terminated")
    truncated_arr = None if truncated is None else _u8_view_from_bool_or_u8(truncated, "truncated")
    if rewards_arr.ndim != 2:
        raise ValueError(f"rewards must have shape (steps, env_count), got {rewards_arr.shape}")
    if values_arr.shape != rewards_arr.shape:
        raise ValueError("values must match rewards shape")
    if terminated_arr.shape != rewards_arr.shape:
        raise ValueError("terminated must match rewards shape")
    if truncated_arr is not None and truncated_arr.shape != rewards_arr.shape:
        raise ValueError("truncated must match rewards shape")

    steps, env_count = rewards_arr.shape
    if steps <= 0 or env_count <= 0:
        raise ValueError("steps and env_count must be positive")
    if bootstrap_values is None:
        bootstrap_ptr = ctypes.POINTER(ctypes.c_float)()
    else:
        bootstrap_arr = np.ascontiguousarray(bootstrap_values, dtype=np.float32)
        if bootstrap_arr.shape != (env_count,):
            raise ValueError(f"bootstrap_values must have shape ({env_count},), got {bootstrap_arr.shape}")
        bootstrap_ptr = bootstrap_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib = _load_library(resolve_library_path(library_path))
    if not hasattr(lib, "trainv3_compute_gae"):
        raise RuntimeError("loaded trainv3_core library does not support Rust GAE computation")

    advantages = np.empty_like(rewards_arr, dtype=np.float32)
    returns = np.empty_like(rewards_arr, dtype=np.float32)
    truncated_ptr = (
        ctypes.POINTER(ctypes.c_uint8)()
        if truncated_arr is None
        else truncated_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    )
    rc = lib.trainv3_compute_gae(
        rewards_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        terminated_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        truncated_ptr,
        bootstrap_ptr,
        ctypes.c_size_t(steps),
        ctypes.c_size_t(env_count),
        ctypes.c_float(float(gamma)),
        ctypes.c_float(float(gae_lambda)),
        ctypes.c_uint8(1 if normalize_advantages else 0),
        advantages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        returns.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"trainv3_compute_gae failed: {rc}")
    return advantages, returns


def compute_rust_selected_local_indices(
    actions,
    legal_action_counts,
    legal_action_offsets,
    legal_action_ids,
    *,
    library_path: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Map selected action ids to local positions inside each legal-action row via Rust FFI."""
    actions_arr = np.ascontiguousarray(actions, dtype=np.uintp).reshape(-1)
    counts_arr = np.ascontiguousarray(legal_action_counts, dtype=np.uintp).reshape(-1)
    offsets_arr = np.ascontiguousarray(legal_action_offsets, dtype=np.uintp).reshape(-1)
    ids_arr = np.ascontiguousarray(legal_action_ids, dtype=np.uintp).reshape(-1)
    row_count = int(actions_arr.shape[0])
    if row_count <= 0:
        raise ValueError("actions must contain at least one row")
    if counts_arr.shape != actions_arr.shape or offsets_arr.shape != actions_arr.shape:
        raise ValueError("actions, legal_action_counts, and legal_action_offsets must have matching flat shapes")

    lib = _load_library(resolve_library_path(library_path))
    if not hasattr(lib, "trainv3_select_local_indices"):
        raise RuntimeError("loaded trainv3_core library does not support Rust selected-index computation")

    selected = np.empty(row_count, dtype=np.int32)
    rc = lib.trainv3_select_local_indices(
        actions_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        counts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        offsets_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(ids_arr.shape[0]),
        ctypes.c_size_t(row_count),
        selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if rc != 0:
        raise RuntimeError(f"trainv3_select_local_indices failed: {rc}")
    return selected


def compute_rust_prepare_ppo_batch(
    rewards,
    values,
    terminated,
    truncated,
    actions,
    legal_action_counts,
    legal_action_offsets,
    legal_action_ids,
    *,
    bootstrap_values=None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
    library_path: str | os.PathLike[str] | None = None,
) -> RustPreparedPPOBatch:
    """Compute GAE/returns and selected-local action indices in one Rust FFI call."""
    rewards_arr = np.ascontiguousarray(rewards, dtype=np.float32)
    values_arr = np.ascontiguousarray(values, dtype=np.float32)
    terminated_arr = _u8_view_from_bool_or_u8(terminated, "terminated")
    truncated_arr = None if truncated is None else _u8_view_from_bool_or_u8(truncated, "truncated")
    actions_arr = np.ascontiguousarray(actions, dtype=np.uintp)
    counts_arr = np.ascontiguousarray(legal_action_counts, dtype=np.uintp)
    offsets_arr = np.ascontiguousarray(legal_action_offsets, dtype=np.uintp)
    ids_arr = np.ascontiguousarray(legal_action_ids, dtype=np.uintp).reshape(-1)
    if rewards_arr.ndim != 2:
        raise ValueError(f"rewards must have shape (steps, env_count), got {rewards_arr.shape}")
    if values_arr.shape != rewards_arr.shape:
        raise ValueError("values must match rewards shape")
    if terminated_arr.shape != rewards_arr.shape:
        raise ValueError("terminated must match rewards shape")
    if truncated_arr is not None and truncated_arr.shape != rewards_arr.shape:
        raise ValueError("truncated must match rewards shape")
    if actions_arr.shape != rewards_arr.shape:
        raise ValueError("actions must match rewards shape")
    if counts_arr.shape != rewards_arr.shape or offsets_arr.shape != rewards_arr.shape:
        raise ValueError("legal_action_counts and legal_action_offsets must match rewards shape")

    steps, env_count = rewards_arr.shape
    if steps <= 0 or env_count <= 0:
        raise ValueError("steps and env_count must be positive")
    if bootstrap_values is None:
        bootstrap_ptr = ctypes.POINTER(ctypes.c_float)()
    else:
        bootstrap_arr = np.ascontiguousarray(bootstrap_values, dtype=np.float32)
        if bootstrap_arr.shape != (env_count,):
            raise ValueError(f"bootstrap_values must have shape ({env_count},), got {bootstrap_arr.shape}")
        bootstrap_ptr = bootstrap_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib = _load_library(resolve_library_path(library_path))
    if not hasattr(lib, "trainv3_prepare_ppo_batch"):
        raise RuntimeError("loaded trainv3_core library does not support fused Rust PPO preparation")

    advantages = np.empty_like(rewards_arr, dtype=np.float32)
    returns = np.empty_like(rewards_arr, dtype=np.float32)
    selected = np.empty(rewards_arr.shape, dtype=np.int32)
    truncated_ptr = (
        ctypes.POINTER(ctypes.c_uint8)()
        if truncated_arr is None
        else truncated_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    )
    rc = lib.trainv3_prepare_ppo_batch(
        rewards_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        terminated_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        truncated_ptr,
        bootstrap_ptr,
        actions_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        counts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        offsets_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(ids_arr.shape[0]),
        ctypes.c_size_t(steps),
        ctypes.c_size_t(env_count),
        ctypes.c_float(float(gamma)),
        ctypes.c_float(float(gae_lambda)),
        ctypes.c_uint8(1 if normalize_advantages else 0),
        advantages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        returns.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if rc != 0:
        raise RuntimeError(f"trainv3_prepare_ppo_batch failed: {rc}")
    return RustPreparedPPOBatch(
        advantages=advantages,
        returns=returns,
        selected_local_indices=selected,
    )


def compute_rust_pad_legal_actions(
    legal_action_counts,
    legal_action_offsets,
    legal_action_ids,
    legal_action_features,
    *,
    library_path: str | os.PathLike[str] | None = None,
) -> RustPaddedLegalActions:
    """Pad compact legal-action ids/features through the Rust FFI kernel."""
    counts_arr = np.ascontiguousarray(legal_action_counts, dtype=np.uintp).reshape(-1)
    offsets_arr = np.ascontiguousarray(legal_action_offsets, dtype=np.uintp).reshape(-1)
    ids_arr = np.ascontiguousarray(legal_action_ids, dtype=np.uintp).reshape(-1)
    features_arr = np.ascontiguousarray(legal_action_features, dtype=np.float32)
    row_count = int(counts_arr.shape[0])
    if row_count <= 0:
        raise ValueError("legal_action_counts must contain at least one row")
    if offsets_arr.shape != counts_arr.shape:
        raise ValueError("legal_action_offsets must match legal_action_counts shape")
    if features_arr.ndim != 2:
        raise ValueError(f"legal_action_features must be 2D, got shape {features_arr.shape}")
    if features_arr.shape[0] != ids_arr.shape[0]:
        raise ValueError("legal_action_features rows must match legal_action_ids length")
    feature_dim = int(features_arr.shape[1])
    if feature_dim <= 0:
        raise ValueError("legal_action_features must have at least one feature column")
    max_legal = int(counts_arr.max(initial=0))
    if max_legal <= 0:
        raise ValueError("at least one legal action row is required")

    lib = _load_library(resolve_library_path(library_path))
    if not hasattr(lib, "trainv3_pad_legal_actions"):
        raise RuntimeError("loaded trainv3_core library does not support Rust legal-action padding")

    padded_ids = np.empty((row_count, max_legal), dtype=np.uintp)
    padded_features = np.empty((row_count, max_legal, feature_dim), dtype=np.float32)
    mask_u8 = np.empty((row_count, max_legal), dtype=np.uint8)
    rc = lib.trainv3_pad_legal_actions(
        counts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        offsets_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(ids_arr.shape[0]),
        features_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(features_arr.size),
        ctypes.c_size_t(row_count),
        ctypes.c_size_t(feature_dim),
        ctypes.c_size_t(max_legal),
        padded_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        padded_features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mask_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if rc != 0:
        raise RuntimeError(f"trainv3_pad_legal_actions failed: {rc}")
    return RustPaddedLegalActions(
        features=padded_features,
        ids=padded_ids,
        mask=_bool_view_from_u8(mask_u8),
    )


def compute_rust_pack_legal_action_rows(
    row_indices,
    legal_action_counts,
    legal_action_offsets,
    legal_action_ids,
    legal_action_features,
    *,
    library_path: str | os.PathLike[str] | None = None,
) -> RustPackedLegalRows:
    """Pack selected variable-length legal-action rows through the Rust FFI kernel."""
    raw_indices = np.asarray(row_indices)
    if raw_indices.ndim != 1:
        raise ValueError(f"row_indices must be 1D, got shape {raw_indices.shape}")
    if raw_indices.size <= 0:
        raise ValueError("row_indices must contain at least one row")
    if not np.issubdtype(raw_indices.dtype, np.unsignedinteger) and np.any(raw_indices < 0):
        raise ValueError("row_indices must be non-negative")

    indices_arr = np.ascontiguousarray(raw_indices, dtype=np.uintp).reshape(-1)
    counts_arr = np.ascontiguousarray(legal_action_counts, dtype=np.uintp).reshape(-1)
    offsets_arr = np.ascontiguousarray(legal_action_offsets, dtype=np.uintp).reshape(-1)
    ids_arr = np.ascontiguousarray(legal_action_ids, dtype=np.uintp).reshape(-1)
    features_arr = np.ascontiguousarray(legal_action_features, dtype=np.float32)
    if counts_arr.shape != offsets_arr.shape:
        raise ValueError("legal_action_counts and legal_action_offsets must have matching flat shapes")
    if counts_arr.size <= 0:
        raise ValueError("legal_action_counts must contain at least one row")
    if int(indices_arr.max(initial=0)) >= int(counts_arr.shape[0]):
        raise IndexError(f"row_indices out of range for {counts_arr.shape[0]} legal rows")
    if features_arr.ndim != 2:
        raise ValueError(f"legal_action_features must be 2D, got shape {features_arr.shape}")
    if features_arr.shape[0] != ids_arr.shape[0]:
        raise ValueError("legal_action_features rows must match legal_action_ids length")
    feature_dim = int(features_arr.shape[1])
    if feature_dim <= 0:
        raise ValueError("legal_action_features must have at least one feature column")

    selected_counts = counts_arr[indices_arr.astype(np.intp, copy=False)]
    if np.any(selected_counts == 0):
        raise ValueError("selected legal-action rows must contain at least one action")
    packed_total = int(selected_counts.sum(dtype=np.uintp))
    if packed_total <= 0:
        raise ValueError("selected legal-action rows must contain at least one action")

    lib = _load_library(resolve_library_path(library_path))
    if not hasattr(lib, "trainv3_pack_legal_action_rows"):
        raise RuntimeError("loaded trainv3_core library does not support Rust legal row packing")

    packed_counts = np.empty(indices_arr.shape[0], dtype=np.uintp)
    packed_offsets = np.empty(indices_arr.shape[0], dtype=np.uintp)
    packed_ids = np.empty(packed_total, dtype=np.uintp)
    packed_features = np.empty((packed_total, feature_dim), dtype=np.float32)
    rc = lib.trainv3_pack_legal_action_rows(
        indices_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(indices_arr.shape[0]),
        counts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(counts_arr.shape[0]),
        offsets_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(offsets_arr.shape[0]),
        ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(ids_arr.shape[0]),
        features_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(features_arr.size),
        ctypes.c_size_t(feature_dim),
        packed_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        packed_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        packed_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(packed_ids.shape[0]),
        packed_features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(packed_features.size),
    )
    if rc != 0:
        raise RuntimeError(f"trainv3_pack_legal_action_rows failed: {rc}")
    return RustPackedLegalRows(
        counts=packed_counts,
        offsets=packed_offsets,
        ids=packed_ids,
        features=packed_features,
    )


def compute_rust_padded_argmax_actions(
    padded_logits,
    legal_action_counts,
    legal_action_ids,
    *,
    library_path: str | os.PathLike[str] | None = None,
) -> RustPaddedArgmaxActions:
    """Select padded legal argmax action ids and log-probs through Rust FFI."""
    logits_arr = np.ascontiguousarray(padded_logits, dtype=np.float32)
    counts_arr = np.ascontiguousarray(legal_action_counts, dtype=np.uintp).reshape(-1)
    ids_arr = np.ascontiguousarray(legal_action_ids, dtype=np.uintp).reshape(-1)
    if logits_arr.ndim != 2:
        raise ValueError(f"padded_logits must be 2D, got shape {logits_arr.shape}")
    row_count, max_legal = (int(logits_arr.shape[0]), int(logits_arr.shape[1]))
    if row_count <= 0 or max_legal <= 0:
        raise ValueError("padded_logits must have non-empty row and legal dimensions")
    if counts_arr.shape != (row_count,):
        raise ValueError(f"legal_action_counts must have shape ({row_count},), got {counts_arr.shape}")
    if np.any(counts_arr == 0):
        raise ValueError("each row must contain at least one legal action")
    if int(counts_arr.max(initial=0)) > max_legal:
        raise ValueError("legal_action_counts cannot exceed padded_logits width")
    if int(counts_arr.sum(dtype=np.uintp)) != int(ids_arr.shape[0]):
        raise ValueError("legal_action_ids length must match legal_action_counts sum")

    lib = _load_library(resolve_library_path(library_path))
    if not hasattr(lib, "trainv3_padded_argmax_actions"):
        raise RuntimeError("loaded trainv3_core library does not support Rust padded argmax selection")

    actions = np.empty(row_count, dtype=np.uintp)
    selected_local = np.empty(row_count, dtype=np.int32)
    log_probs = np.empty(row_count, dtype=np.float32)
    rc = lib.trainv3_padded_argmax_actions(
        logits_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(logits_arr.size),
        counts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(ids_arr.shape[0]),
        ctypes.c_size_t(row_count),
        ctypes.c_size_t(max_legal),
        actions.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        selected_local.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        log_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"trainv3_padded_argmax_actions failed: {rc}")
    return RustPaddedArgmaxActions(
        actions=actions,
        selected_local_indices=selected_local,
        log_probs=log_probs,
    )


def compute_rust_compact_argmax_actions(
    legal_logits,
    legal_action_counts,
    legal_action_ids,
    *,
    library_path: str | os.PathLike[str] | None = None,
) -> RustCompactArgmaxActions:
    """Select flat compact legal argmax action ids and log-probs through Rust FFI."""
    logits_arr = np.ascontiguousarray(legal_logits, dtype=np.float32).reshape(-1)
    counts_arr = np.ascontiguousarray(legal_action_counts, dtype=np.uintp).reshape(-1)
    ids_arr = np.ascontiguousarray(legal_action_ids, dtype=np.uintp).reshape(-1)
    row_count = int(counts_arr.shape[0])
    if row_count <= 0:
        raise ValueError("legal_action_counts must contain at least one row")
    if np.any(counts_arr == 0):
        raise ValueError("each row must contain at least one legal action")
    if int(counts_arr.sum(dtype=np.uintp)) != int(logits_arr.shape[0]):
        raise ValueError("legal_logits length must match legal_action_counts sum")
    if ids_arr.shape != logits_arr.shape:
        raise ValueError("legal_action_ids length must match legal_logits length")

    lib = _load_library(resolve_library_path(library_path))
    if not hasattr(lib, "trainv3_compact_argmax_actions"):
        raise RuntimeError("loaded trainv3_core library does not support Rust compact argmax selection")

    actions = np.empty(row_count, dtype=np.uintp)
    selected_local = np.empty(row_count, dtype=np.int32)
    log_probs = np.empty(row_count, dtype=np.float32)
    rc = lib.trainv3_compact_argmax_actions(
        logits_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(logits_arr.shape[0]),
        counts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(row_count),
        ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(ids_arr.shape[0]),
        actions.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        selected_local.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        log_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"trainv3_compact_argmax_actions failed: {rc}")
    return RustCompactArgmaxActions(
        actions=actions,
        selected_local_indices=selected_local,
        log_probs=log_probs,
    )


def compute_rust_dense_argmax_actions(
    logits,
    action_mask,
    *,
    library_path: str | os.PathLike[str] | None = None,
) -> RustDenseArgmaxActions:
    """Select dense masked argmax action ids and log-probs through Rust FFI."""
    logits_arr = np.ascontiguousarray(logits, dtype=np.float32)
    mask_arr = _dense_mask_u8_view_or_copy(action_mask)
    if logits_arr.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {logits_arr.shape}")
    if mask_arr.shape != logits_arr.shape:
        raise ValueError(f"action_mask shape {mask_arr.shape} must match logits shape {logits_arr.shape}")
    row_count, action_count = (int(logits_arr.shape[0]), int(logits_arr.shape[1]))
    if row_count <= 0 or action_count <= 0:
        raise ValueError("logits must have non-empty row and action dimensions")
    if np.any(mask_arr.sum(axis=1) == 0):
        raise ValueError("each row must contain at least one legal action")

    lib = _load_library(resolve_library_path(library_path))
    if not hasattr(lib, "trainv3_dense_argmax_actions"):
        raise RuntimeError("loaded trainv3_core library does not support Rust dense argmax selection")

    actions = np.empty(row_count, dtype=np.uintp)
    log_probs = np.empty(row_count, dtype=np.float32)
    rc = lib.trainv3_dense_argmax_actions(
        logits_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(logits_arr.size),
        mask_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_size_t(mask_arr.size),
        ctypes.c_size_t(row_count),
        ctypes.c_size_t(action_count),
        actions.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        log_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"trainv3_dense_argmax_actions failed: {rc}")
    return RustDenseArgmaxActions(actions=actions, log_probs=log_probs)


def compute_rust_repeat_row_indices(
    legal_action_counts,
    *,
    library_path: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Expand compact legal-action counts into flat row indices through Rust FFI."""
    counts_arr = np.ascontiguousarray(legal_action_counts, dtype=np.uintp).reshape(-1)
    row_count = int(counts_arr.shape[0])
    if row_count <= 0:
        raise ValueError("legal_action_counts must contain at least one row")
    if np.any(counts_arr == 0):
        raise ValueError("each row must contain at least one legal action")
    total = int(counts_arr.sum(dtype=np.uintp))
    if total <= 0:
        raise ValueError("legal_action_counts sum must be positive")

    lib = _load_library(resolve_library_path(library_path))
    if not hasattr(lib, "trainv3_repeat_row_indices"):
        raise RuntimeError("loaded trainv3_core library does not support Rust row-index expansion")

    row_indices = np.empty(total, dtype=np.int32)
    rc = lib.trainv3_repeat_row_indices(
        counts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(row_count),
        row_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(row_indices.shape[0]),
    )
    if rc != 0:
        raise RuntimeError(f"trainv3_repeat_row_indices failed: {rc}")
    return row_indices


def compute_rust_normalized_legal_offsets(
    legal_action_counts,
    legal_action_offsets,
    *,
    library_path: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Shift contiguous legal-action offsets to a zero-based slice through Rust FFI."""
    counts_arr = np.ascontiguousarray(legal_action_counts, dtype=np.uintp).reshape(-1)
    offsets_arr = np.ascontiguousarray(legal_action_offsets, dtype=np.uintp).reshape(-1)
    row_count = int(counts_arr.shape[0])
    if row_count <= 0:
        raise ValueError("legal_action_counts must contain at least one row")
    if offsets_arr.shape != counts_arr.shape:
        raise ValueError("legal_action_counts and legal_action_offsets must have matching flat shapes")
    if np.any(counts_arr == 0):
        raise ValueError("each row must contain at least one legal action")

    lib = _load_library(resolve_library_path(library_path))
    if not hasattr(lib, "trainv3_normalize_legal_offsets"):
        raise RuntimeError("loaded trainv3_core library does not support Rust legal offset normalization")

    normalized = np.empty(row_count, dtype=np.uintp)
    rc = lib.trainv3_normalize_legal_offsets(
        counts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        offsets_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(row_count),
        normalized.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
    )
    if rc != 0:
        raise RuntimeError(f"trainv3_normalize_legal_offsets failed: {rc}")
    return normalized


class RustBatchWorker:
    """Python-facing wrapper around the Rust batched rollout worker.

    Returned arrays are views over Rust-owned buffers by default. Copy them before
    the next `step()`/`reset()`/`encode()` call if they need to outlive that call.
    """

    def __init__(
        self,
        ptr: int,
        lib: ctypes.CDLL,
        env_count: int,
        *,
        action_features_dtype: str = "float32",
        action_features_mode: str = "dense_and_legal",
        reset_pool_mode: str = "fixed",
        observation_mode: str = "v1_and_v5",
        action_mask_mode: str = "dense",
        terminal_observation_mode: str = "full",
        diagnostic_mode: str = "full",
    ):
        if not ptr:
            raise RuntimeError("failed to create Rust TrainV3 worker")
        self._ptr = ctypes.c_void_p(ptr)
        self._lib = lib
        self.env_count = int(env_count)
        self.action_features_dtype = _normalize_action_features_dtype(action_features_dtype)
        self.action_features_mode = _normalize_action_features_mode(action_features_mode)
        self.reset_pool_mode = _normalize_reset_pool_mode(reset_pool_mode)
        self.observation_mode = _normalize_observation_mode(observation_mode)
        self.action_mask_mode = _normalize_action_mask_mode(action_mask_mode)
        self.terminal_observation_mode = _normalize_terminal_observation_mode(terminal_observation_mode)
        self.diagnostic_mode = _normalize_diagnostic_mode(diagnostic_mode)

    @classmethod
    def from_trace_file(
        cls,
        path: str | os.PathLike[str],
        *,
        env_count: int,
        library_path: str | os.PathLike[str] | None = None,
        action_features_dtype: str = "float32",
        action_features_mode: str = "dense_and_legal",
        observation_mode: str = "v1_and_v5",
        action_mask_mode: str = "dense",
        terminal_observation_mode: str = "full",
        diagnostic_mode: str = "full",
    ) -> "RustBatchWorker":
        action_features_dtype = _normalize_action_features_dtype(action_features_dtype)
        action_features_mode = _normalize_action_features_mode(action_features_mode)
        observation_mode = _normalize_observation_mode(observation_mode)
        action_mask_mode = _normalize_action_mask_mode(action_mask_mode)
        terminal_observation_mode = _normalize_terminal_observation_mode(terminal_observation_mode)
        diagnostic_mode = _normalize_diagnostic_mode(diagnostic_mode)
        raw = Path(path).read_bytes()
        lib = _load_library(resolve_library_path(library_path))
        buf = ctypes.create_string_buffer(raw)
        dtype_code = 1 if action_features_dtype == "float16" else 0
        output_code = 1 if action_features_mode == "legal_only" else 0
        observation_code = 1 if observation_mode == "v5_only" else 0
        action_mask_code = 1 if action_mask_mode == "legal_only" else 0
        terminal_observation_code = 1 if terminal_observation_mode == "none" else 0
        diagnostic_code = 1 if diagnostic_mode == "none" else 0
        if hasattr(lib, "trainv3_worker_from_trace_json_with_options_v6"):
            ptr = lib.trainv3_worker_from_trace_json_with_options_v6(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
                ctypes.c_uint32(observation_code),
                ctypes.c_uint32(action_mask_code),
                ctypes.c_uint32(terminal_observation_code),
                ctypes.c_uint32(diagnostic_code),
            )
        elif diagnostic_mode != "full":
            raise RuntimeError("loaded trainv3_core library does not support diagnostic output modes")
        elif hasattr(lib, "trainv3_worker_from_trace_json_with_options_v5"):
            ptr = lib.trainv3_worker_from_trace_json_with_options_v5(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
                ctypes.c_uint32(observation_code),
                ctypes.c_uint32(action_mask_code),
                ctypes.c_uint32(terminal_observation_code),
            )
        elif terminal_observation_mode != "full":
            raise RuntimeError("loaded trainv3_core library does not support terminal observation output modes")
        elif hasattr(lib, "trainv3_worker_from_trace_json_with_options_v4"):
            ptr = lib.trainv3_worker_from_trace_json_with_options_v4(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
                ctypes.c_uint32(observation_code),
                ctypes.c_uint32(action_mask_code),
            )
        elif action_mask_mode != "dense":
            raise RuntimeError("loaded trainv3_core library does not support action mask output modes")
        elif hasattr(lib, "trainv3_worker_from_trace_json_with_options_v3"):
            ptr = lib.trainv3_worker_from_trace_json_with_options_v3(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
                ctypes.c_uint32(observation_code),
            )
        elif observation_mode != "v1_and_v5":
            raise RuntimeError("loaded trainv3_core library does not support observation output modes")
        elif hasattr(lib, "trainv3_worker_from_trace_json_with_options_v2"):
            ptr = lib.trainv3_worker_from_trace_json_with_options_v2(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
            )
        elif action_features_mode == "legal_only":
            raise RuntimeError("loaded trainv3_core library does not support legal_only action feature output")
        elif action_features_dtype == "float16":
            ptr = lib.trainv3_worker_from_trace_json_with_options(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
            )
        else:
            ptr = lib.trainv3_worker_from_trace_json(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
            )
        return cls(
            ptr,
            lib,
            env_count,
            action_features_dtype=action_features_dtype,
            action_features_mode=action_features_mode,
            reset_pool_mode="fixed",
            observation_mode=observation_mode,
            action_mask_mode=action_mask_mode,
            terminal_observation_mode=terminal_observation_mode,
            diagnostic_mode=diagnostic_mode,
        )

    @classmethod
    def from_trace_files(
        cls,
        paths: list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...],
        *,
        env_count: int,
        library_path: str | os.PathLike[str] | None = None,
        action_features_dtype: str = "float32",
        action_features_mode: str = "dense_and_legal",
        reset_pool_mode: str = "fixed",
        observation_mode: str = "v1_and_v5",
        action_mask_mode: str = "dense",
        terminal_observation_mode: str = "full",
        diagnostic_mode: str = "full",
    ) -> "RustBatchWorker":
        action_features_dtype = _normalize_action_features_dtype(action_features_dtype)
        action_features_mode = _normalize_action_features_mode(action_features_mode)
        reset_pool_mode = _normalize_reset_pool_mode(reset_pool_mode)
        observation_mode = _normalize_observation_mode(observation_mode)
        action_mask_mode = _normalize_action_mask_mode(action_mask_mode)
        terminal_observation_mode = _normalize_terminal_observation_mode(terminal_observation_mode)
        diagnostic_mode = _normalize_diagnostic_mode(diagnostic_mode)
        raw_paths = [Path(path) for path in paths]
        if not raw_paths:
            raise ValueError("paths must contain at least one trace file")
        raw = b"[" + b",".join(path.read_bytes() for path in raw_paths) + b"]"
        lib = _load_library(resolve_library_path(library_path))
        if not (
            hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v3")
            or hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v2")
        ):
            raise RuntimeError("loaded trainv3_core library does not support trace-file pools")
        if (
            reset_pool_mode != "fixed"
            and not hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v3")
        ):
            raise RuntimeError("loaded trainv3_core library does not support trace-pool reset modes")
        if (
            observation_mode != "v1_and_v5"
            and not hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v4")
        ):
            raise RuntimeError("loaded trainv3_core library does not support observation output modes")
        if (
            action_mask_mode != "dense"
            and not hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v5")
        ):
            raise RuntimeError("loaded trainv3_core library does not support action mask output modes")
        if (
            terminal_observation_mode != "full"
            and not hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v6")
        ):
            raise RuntimeError("loaded trainv3_core library does not support terminal observation output modes")
        if (
            diagnostic_mode != "full"
            and not hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v7")
        ):
            raise RuntimeError("loaded trainv3_core library does not support diagnostic output modes")
        buf = ctypes.create_string_buffer(raw)
        dtype_code = 1 if action_features_dtype == "float16" else 0
        output_code = 1 if action_features_mode == "legal_only" else 0
        reset_pool_mode_code = 1 if reset_pool_mode == "cycle" else 0
        observation_code = 1 if observation_mode == "v5_only" else 0
        action_mask_code = 1 if action_mask_mode == "legal_only" else 0
        terminal_observation_code = 1 if terminal_observation_mode == "none" else 0
        diagnostic_code = 1 if diagnostic_mode == "none" else 0
        if hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v7"):
            ptr = lib.trainv3_worker_from_trace_json_pool_with_options_v7(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
                ctypes.c_uint32(reset_pool_mode_code),
                ctypes.c_uint32(observation_code),
                ctypes.c_uint32(action_mask_code),
                ctypes.c_uint32(terminal_observation_code),
                ctypes.c_uint32(diagnostic_code),
            )
        elif hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v6"):
            ptr = lib.trainv3_worker_from_trace_json_pool_with_options_v6(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
                ctypes.c_uint32(reset_pool_mode_code),
                ctypes.c_uint32(observation_code),
                ctypes.c_uint32(action_mask_code),
                ctypes.c_uint32(terminal_observation_code),
            )
        elif hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v5"):
            ptr = lib.trainv3_worker_from_trace_json_pool_with_options_v5(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
                ctypes.c_uint32(reset_pool_mode_code),
                ctypes.c_uint32(observation_code),
                ctypes.c_uint32(action_mask_code),
            )
        elif hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v4"):
            ptr = lib.trainv3_worker_from_trace_json_pool_with_options_v4(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
                ctypes.c_uint32(reset_pool_mode_code),
                ctypes.c_uint32(observation_code),
            )
        elif hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v3"):
            ptr = lib.trainv3_worker_from_trace_json_pool_with_options_v3(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
                ctypes.c_uint32(reset_pool_mode_code),
            )
        else:
            ptr = lib.trainv3_worker_from_trace_json_pool_with_options_v2(
                ctypes.cast(buf, ctypes.c_void_p),
                ctypes.c_size_t(len(raw)),
                ctypes.c_size_t(env_count),
                ctypes.c_uint32(dtype_code),
                ctypes.c_uint32(output_code),
            )
        if not ptr:
            raise ValueError(
                "failed to create Rust TrainV3 trace pool worker; shared settings "
                "placement_mode, mana_per_turn, and v5_weighted_reward must match"
            )
        return cls(
            ptr,
            lib,
            env_count,
            action_features_dtype=action_features_dtype,
            action_features_mode=action_features_mode,
            reset_pool_mode=reset_pool_mode,
            observation_mode=observation_mode,
            action_mask_mode=action_mask_mode,
            terminal_observation_mode=terminal_observation_mode,
            diagnostic_mode=diagnostic_mode,
        )

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            self._lib.trainv3_worker_free(self._ptr)
            self._ptr = None

    def __enter__(self) -> "RustBatchWorker":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def encode(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        rc = self._lib.trainv3_worker_encode(self._nonnull_ptr())
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_encode failed: {rc}")
        return self.arrays(copy=copy)

    def reset(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        rc = self._lib.trainv3_worker_reset(self._nonnull_ptr())
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_reset failed: {rc}")
        return self.arrays(copy=copy)

    def use_chacha_rng(self) -> None:
        if not hasattr(self._lib, "trainv3_worker_use_chacha_rng"):
            raise RuntimeError("loaded trainv3_core library does not support live ChaCha RNG")
        rc = self._lib.trainv3_worker_use_chacha_rng(self._nonnull_ptr())
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_use_chacha_rng failed: {rc}")

    def reset_indices(self, indices, *, copy: bool = False) -> dict[str, np.ndarray]:
        reset_indices = np.ascontiguousarray(indices, dtype=np.uintp)
        if reset_indices.ndim != 1:
            raise ValueError(f"expected reset indices to be 1D, got shape {reset_indices.shape}")
        ptr = reset_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        rc = self._lib.trainv3_worker_reset_indices(
            self._nonnull_ptr(),
            ptr,
            ctypes.c_size_t(reset_indices.size),
        )
        if rc == -3:
            raise ValueError("trainv3_worker_reset_indices failed: index out of bounds")
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_reset_indices failed: {rc}")
        return self.arrays(copy=copy)

    def step(self, action_ids, *, copy: bool = False) -> dict[str, np.ndarray]:
        actions = np.ascontiguousarray(action_ids, dtype=np.uintp)
        if actions.shape != (self.env_count,):
            raise ValueError(f"expected action_ids shape ({self.env_count},), got {actions.shape}")
        ptr = actions.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        rc = self._lib.trainv3_worker_step(self._nonnull_ptr(), ptr, ctypes.c_size_t(actions.size))
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_step failed: {rc}")
        return self.arrays(copy=copy)

    def step_auto_reset(self, action_ids, *, copy: bool = False) -> dict[str, np.ndarray]:
        actions = np.ascontiguousarray(action_ids, dtype=np.uintp)
        if actions.shape != (self.env_count,):
            raise ValueError(f"expected action_ids shape ({self.env_count},), got {actions.shape}")
        ptr = actions.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        rc = self._lib.trainv3_worker_step_auto_reset(self._nonnull_ptr(), ptr, ctypes.c_size_t(actions.size))
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_step_auto_reset failed: {rc}")
        return self.arrays(copy=copy)

    def current_actor_ids(self) -> np.ndarray:
        if not hasattr(self._lib, "trainv3_worker_current_actor_ids"):
            raise RuntimeError("loaded trainv3_core library does not expose current actor ids")
        out = np.empty((self.env_count,), dtype=np.int32)
        rc = self._lib.trainv3_worker_current_actor_ids(
            self._nonnull_ptr(),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_size_t(out.size),
        )
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_current_actor_ids failed: {rc}")
        return out

    def select_rule_actions(self, agent_codes, *, salt: int = 0) -> np.ndarray:
        if not hasattr(self._lib, "trainv3_worker_select_rule_actions"):
            raise RuntimeError("loaded trainv3_core library does not expose rule-action selection")
        codes = np.ascontiguousarray(agent_codes, dtype=np.uint32)
        if codes.shape != (self.env_count,):
            raise ValueError(f"expected agent_codes shape ({self.env_count},), got {codes.shape}")
        out = np.empty((self.env_count,), dtype=np.uintp)
        rc = self._lib.trainv3_worker_select_rule_actions(
            self._nonnull_ptr(),
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_size_t(codes.size),
            ctypes.c_uint64(int(salt) & ((1 << 64) - 1)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            ctypes.c_size_t(out.size),
        )
        if rc == -4:
            raise ValueError("trainv3_worker_select_rule_actions failed: invalid agent code or no legal action")
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_select_rule_actions failed: {rc}")
        return out

    def advance_rule_until_actor(
        self,
        learner_actor_ids,
        agent_codes,
        *,
        max_actions_per_env: int = 64,
        salt: int = 0,
        auto_reset: bool = True,
        copy: bool = False,
    ) -> dict[str, np.ndarray]:
        if not hasattr(self._lib, "trainv3_worker_advance_rule_until_actor"):
            raise RuntimeError("loaded trainv3_core library does not expose rule-opponent fast-forward")
        learners = np.ascontiguousarray(learner_actor_ids, dtype=np.int32)
        codes = np.ascontiguousarray(agent_codes, dtype=np.uint32)
        if learners.shape != (self.env_count,):
            raise ValueError(f"expected learner_actor_ids shape ({self.env_count},), got {learners.shape}")
        if codes.shape != (self.env_count,):
            raise ValueError(f"expected agent_codes shape ({self.env_count},), got {codes.shape}")
        rewards = np.empty((self.env_count,), dtype=np.float32)
        terminated_u8 = np.empty((self.env_count,), dtype=np.uint8)
        truncated_u8 = np.empty((self.env_count,), dtype=np.uint8)
        reset_u8 = np.empty((self.env_count,), dtype=np.uint8)
        counts = np.empty((self.env_count,), dtype=np.uintp)
        rc = self._lib.trainv3_worker_advance_rule_until_actor(
            self._nonnull_ptr(),
            learners.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_size_t(learners.size),
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_size_t(codes.size),
            ctypes.c_size_t(int(max_actions_per_env)),
            ctypes.c_uint64(int(salt) & ((1 << 64) - 1)),
            ctypes.c_uint8(1 if auto_reset else 0),
            rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(rewards.size),
            terminated_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(terminated_u8.size),
            truncated_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(truncated_u8.size),
            reset_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(reset_u8.size),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            ctypes.c_size_t(counts.size),
        )
        if rc == -5:
            raise ValueError("trainv3_worker_advance_rule_until_actor failed: invalid rule fast-forward")
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_advance_rule_until_actor failed: {rc}")
        raw = self.arrays(copy=copy)
        raw["rule_learner_rewards"] = rewards
        raw["rule_terminated"] = _bool_view_from_u8(terminated_u8)
        raw["rule_truncated"] = _bool_view_from_u8(truncated_u8)
        raw["rule_reset_flags"] = _bool_view_from_u8(reset_u8)
        raw["rule_action_counts"] = counts
        return raw

    def truncated(self) -> np.ndarray:
        """Per-env truncation flag (WD-2): ``turn_number > max_turns``.

        Additive accessor — the underlying ``trainv3_worker_truncated_ptr/_len``
        FFI exists and is populated every ``set_last``; ``arrays()`` simply omits
        it. The A4 live-self-play trainer reads this to detect ``max_turns``
        truncation on the LIVE path (kernel.rs:807 ``truncated = next.turn_number
        > self.config.max_turns``).
        """
        if not hasattr(self._lib, "trainv3_worker_truncated_ptr"):
            raise RuntimeError("loaded trainv3_core library does not expose truncated flag")
        u8 = self._u8_array(
            self._lib.trainv3_worker_truncated_ptr,
            self._lib.trainv3_worker_truncated_len,
            (self.env_count,),
        )
        return _bool_view_from_u8(u8)

    def mana_draw_legal(self) -> np.ndarray:
        """Per-env parallel mana_draw-head legality flag (Phase 2: MD-3).

        Additive accessor — ``trainv3_worker_mana_draw_legal_ptr/_len`` FFI exists
        (populated every ``set_last`` from ``BatchTensorOutput.mana_draw_legal``).
        The A4 learner policy gates the mana_draw head by this flag and selects
        via ``mana_draw_head_v5.select_includes_mana_draw``.
        """
        if not hasattr(self._lib, "trainv3_worker_mana_draw_legal_ptr"):
            raise RuntimeError("loaded trainv3_core library does not expose mana_draw_legal flag")
        u8 = self._u8_array(
            self._lib.trainv3_worker_mana_draw_legal_ptr,
            self._lib.trainv3_worker_mana_draw_legal_len,
            (self.env_count,),
        )
        return _bool_view_from_u8(u8)

    def step_mana_draw(
        self,
        action_ids,
        mana_draw_flags,
        *,
        copy: bool = False,
    ) -> dict[str, np.ndarray]:
        """Step the batch with a parallel mana_draw flag per env (MD-FFI).

        Composes the existing ``trainv3_worker_step_mana_draw`` FFI
        (ffi.rs:1658) + ``BatchedRolloutWorker::step_with_mana_draw``
        (worker.rs:739). ``mana_draw_flags[i]`` true -> env i applies a mana_draw
        (standalone action that REPLACES the action_id decode, kernel.rs:788) —
        ``action_ids[i]`` is then a placeholder. False -> behaves like ``step``.
        """
        if not hasattr(self._lib, "trainv3_worker_step_mana_draw"):
            raise RuntimeError("loaded trainv3_core library does not expose mana_draw step")
        actions = np.ascontiguousarray(action_ids, dtype=np.uintp)
        if actions.shape != (self.env_count,):
            raise ValueError(f"expected action_ids shape ({self.env_count},), got {actions.shape}")
        flags = np.ascontiguousarray(
            np.fromiter((1 if bool(f) else 0 for f in mana_draw_flags), dtype=np.uint8),
        )
        if flags.shape != (self.env_count,):
            raise ValueError(f"expected mana_draw_flags shape ({self.env_count},), got {flags.shape}")
        rc = self._lib.trainv3_worker_step_mana_draw(
            self._nonnull_ptr(),
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            ctypes.c_size_t(actions.size),
            flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(flags.size),
        )
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_step_mana_draw failed: {rc}")
        return self.arrays(copy=copy)

    def hero_hp(self) -> np.ndarray:
        """Per-env hero hp snapshot ``(env_count, 4)``: ``[p1_hp, p1_max_hp,
        p2_hp, p2_max_hp]`` per env.

        Additive accessor over the new ``trainv3_worker_hero_hp`` FFI
        (mirrors ``current_actor_ids``). Reads the existing per-env
        ``KernelState`` (worker.rs ``states: Vec<KernelState>``). The A4
        live-self-play trainer feeds this to
        ``ppo_phaseA_config.is_decisive_state`` for decisive-early-end (D-A6).
        """
        if not hasattr(self._lib, "trainv3_worker_hero_hp"):
            raise RuntimeError("loaded trainv3_core library does not expose hero_hp accessor")
        out = np.empty((self.env_count, 4), dtype=np.int32)
        rc = self._lib.trainv3_worker_hero_hp(
            self._nonnull_ptr(),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_size_t(out.size),
        )
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_hero_hp failed: {rc}")
        return out

    @classmethod
    def from_live(
        cls,
        *,
        seed: int,
        env_count: int,
        max_turns: int = 120,
        library_path: str | os.PathLike[str] | None = None,
        info_mode: Any | None = None,
        assist_mode: Any | None = None,
        placement_mode: str = "append_only",
        verify_mask: bool = False,
        p1_deck_ids: list[int] | None = None,
        p2_deck_ids: list[int] | None = None,
        action_features_dtype: str = "float32",
        action_features_mode: str = "legal_only",
        observation_mode: str = "v5_only",
        action_mask_mode: str = "legal_only",
        terminal_observation_mode: str = "none",
        diagnostic_mode: str = "none",
    ) -> "RustBatchWorker":
        """THE LIVE-SELF-PLAY CONSTRUCTOR (A4, D-A8 = build live).

        Composes the EXISTING ``from_trace_file`` FFI
        (``trainv3_worker_from_trace_json_with_options_v6``, ffi.rs:703) with an
        init-only ``GoldenTrace`` JSON built by ``golden_trace.build_golden_trace``
        (``steps=0`` -> no turn history, only the initial snapshot). The Rust
        worker steps LIVE from ``trace.initial`` (``trace.steps`` is unused by
        the worker, ffi.rs:744-754), so this yields a live arena — NOT a trace
        replay.

        ``max_turns`` is threaded into ``trace['env_config']['max_turns']`` BEFORE
        the worker is constructed, so ``KernelConfig::from_trace_config``
        (kernel.rs:660) reads the Phase-A value (NOT the serde default 80,
        kernel.rs:624). This is the LIVE-constructor plumbing Fix #2
        (``ppo_phaseA_config.LIVE_MAX_TURNS_THREADING_NOTE``).

        Additive — does NOT modify ``from_trace_file``/``from_trace_files``
        (frozen-classic guard: ``classic_rl_env.py`` untouched; ``rust_ffi.py``
        is NOT frozen-classic).
        """
        from .golden_trace import build_golden_trace  # lazy: keeps rust_ffi import-light
        from .contracts import InfoModeV5, AssistModeV5

        im = info_mode if info_mode is not None else InfoModeV5(
            enemy_hand_known=True,
            enemy_deck_known=True,
            enemy_deck_order_known=True,
        )
        am = assist_mode if assist_mode is not None else AssistModeV5()
        import json as _json
        import tempfile as _tempfile

        paths: list[str] = []
        try:
            # The Rust trace-pool constructor accepts the seed through a
            # platform-sized integer path. League gates derive seeds by
            # multiplying the run seed, so constrain every live trace to the
            # stable positive range before serializing it. This keeps the
            # deterministic per-env spacing while avoiding an opaque FFI
            # construction failure for large gate/tournament seeds.
            seed_modulus = (2**31) - 1
            base_seed = int(seed) % seed_modulus
            for idx in range(int(env_count)):
                trace_seed = (base_seed + idx * 9973) % seed_modulus
                starting_player_id = 1 if idx % 2 == 0 else 2
                trace = build_golden_trace(
                    seed=trace_seed,
                    steps=0,
                    placement_mode=placement_mode,
                    verify_mask=verify_mask,
                    info_mode=im,
                    assist_mode=am,
                    choose="first",
                    p1_deck_ids=p1_deck_ids,
                    p2_deck_ids=p2_deck_ids,
                    max_turns=int(max_turns),
                    starting_player_id=starting_player_id,
                )
                # Defensive: assert max_turns threading before the FFI build (the Rust
                # worker reads trace.env_config.max_turns at kernel.rs:660).
                if int(trace["env_config"].get("max_turns", 0)) != int(max_turns):
                    raise RuntimeError(
                        f"from_live max_turns threading failed: env_config.max_turns="
                        f"{trace['env_config'].get('max_turns')} != {max_turns}"
                    )
                fd, path = _tempfile.mkstemp(suffix=".json", prefix="trainv3_live_")
                paths.append(path)
                with os.fdopen(fd, "w") as fh:
                    _json.dump(trace, fh)
            worker = cls.from_trace_files(
                paths,
                env_count=env_count,
                library_path=library_path,
                action_features_dtype=action_features_dtype,
                action_features_mode=action_features_mode,
                reset_pool_mode="cycle",
                observation_mode=observation_mode,
                action_mask_mode=action_mask_mode,
                terminal_observation_mode=terminal_observation_mode,
                diagnostic_mode=diagnostic_mode,
            )
            worker.use_chacha_rng()
            return worker
        finally:
            for path in paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def rollout_action_tape(self, action_ids, *, auto_reset: bool = False, copy: bool = False) -> dict[str, np.ndarray]:
        actions = np.ascontiguousarray(action_ids, dtype=np.uintp)
        if actions.ndim == 1:
            return self.rollout_broadcast_action_ids(actions, auto_reset=auto_reset, copy=copy)
        if not hasattr(self._lib, "trainv3_worker_rollout_action_tape"):
            raise RuntimeError("loaded trainv3_core library does not support action-tape rollouts")
        if actions.ndim != 2 or actions.shape[1] != self.env_count:
            raise ValueError(
                f"expected action_ids shape (steps, {self.env_count}), got {actions.shape}"
            )
        steps = int(actions.shape[0])
        if steps <= 0:
            raise ValueError("action_ids must contain at least one step")
        ptr = actions.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        rc = self._lib.trainv3_worker_rollout_action_tape(
            self._nonnull_ptr(),
            ptr,
            ctypes.c_size_t(actions.size),
            ctypes.c_size_t(steps),
            ctypes.c_uint8(1 if auto_reset else 0),
        )
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_rollout_action_tape failed: {rc}")
        rows = steps * self.env_count
        raw = self._arrays_for_rows(rows, copy=copy)
        return self._reshape_rollout_arrays(raw, steps=steps)

    def rollout_broadcast_action_ids(
        self,
        action_ids,
        *,
        auto_reset: bool = False,
        copy: bool = False,
    ) -> dict[str, np.ndarray]:
        if not hasattr(self._lib, "trainv3_worker_rollout_broadcast_action_ids"):
            raise RuntimeError("loaded trainv3_core library does not support broadcast action-id rollouts")
        actions = np.ascontiguousarray(action_ids, dtype=np.uintp)
        if actions.ndim != 1:
            raise ValueError(f"expected action_ids shape (steps,), got {actions.shape}")
        steps = int(actions.shape[0])
        if steps <= 0:
            raise ValueError("action_ids must contain at least one step")
        ptr = actions.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        rc = self._lib.trainv3_worker_rollout_broadcast_action_ids(
            self._nonnull_ptr(),
            ptr,
            ctypes.c_size_t(actions.size),
            ctypes.c_uint8(1 if auto_reset else 0),
        )
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_rollout_broadcast_action_ids failed: {rc}")
        rows = steps * self.env_count
        raw = self._arrays_for_rows(rows, copy=copy)
        return self._reshape_rollout_arrays(raw, steps=steps)

    def rollout_action_tape_pre_step(
        self,
        action_ids,
        *,
        auto_reset: bool = False,
        copy: bool = False,
    ) -> dict[str, np.ndarray]:
        actions = np.ascontiguousarray(action_ids, dtype=np.uintp)
        if actions.ndim == 1:
            return self.rollout_broadcast_action_ids_pre_step(actions, auto_reset=auto_reset, copy=copy)
        if not hasattr(self._lib, "trainv3_worker_rollout_action_tape_pre_step"):
            raise RuntimeError("loaded trainv3_core library does not support pre-step action-tape rollouts")
        if actions.ndim != 2 or actions.shape[1] != self.env_count:
            raise ValueError(
                f"expected action_ids shape (steps, {self.env_count}), got {actions.shape}"
            )
        steps = int(actions.shape[0])
        if steps <= 0:
            raise ValueError("action_ids must contain at least one step")
        ptr = actions.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        rc = self._lib.trainv3_worker_rollout_action_tape_pre_step(
            self._nonnull_ptr(),
            ptr,
            ctypes.c_size_t(actions.size),
            ctypes.c_size_t(steps),
            ctypes.c_uint8(1 if auto_reset else 0),
        )
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_rollout_action_tape_pre_step failed: {rc}")
        rows = steps * self.env_count
        raw = self._arrays_for_rows(rows, copy=copy)
        return self._reshape_rollout_arrays(raw, steps=steps)

    def rollout_broadcast_action_ids_pre_step(
        self,
        action_ids,
        *,
        auto_reset: bool = False,
        copy: bool = False,
    ) -> dict[str, np.ndarray]:
        if not hasattr(self._lib, "trainv3_worker_rollout_broadcast_action_ids_pre_step"):
            raise RuntimeError("loaded trainv3_core library does not support pre-step broadcast action-id rollouts")
        actions = np.ascontiguousarray(action_ids, dtype=np.uintp)
        if actions.ndim != 1:
            raise ValueError(f"expected action_ids shape (steps,), got {actions.shape}")
        steps = int(actions.shape[0])
        if steps <= 0:
            raise ValueError("action_ids must contain at least one step")
        ptr = actions.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        rc = self._lib.trainv3_worker_rollout_broadcast_action_ids_pre_step(
            self._nonnull_ptr(),
            ptr,
            ctypes.c_size_t(actions.size),
            ctypes.c_uint8(1 if auto_reset else 0),
        )
        if rc != 0:
            raise RuntimeError(f"trainv3_worker_rollout_broadcast_action_ids_pre_step failed: {rc}")
        rows = steps * self.env_count
        raw = self._arrays_for_rows(rows, copy=copy)
        return self._reshape_rollout_arrays(raw, steps=steps)

    def arrays(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        return self._arrays_for_rows(self.env_count, copy=copy)

    def _arrays_for_rows(self, row_count: int, *, copy: bool = False) -> dict[str, np.ndarray]:
        obs_v1 = self._optional_float_array(
            self._lib.trainv3_worker_observation_v1_ptr,
            self._lib.trainv3_worker_observation_v1_len,
            (row_count, OBS_V1_DIM),
            optional=self.observation_mode == "v5_only",
        )
        obs_v5 = self._float_array(
            self._lib.trainv3_worker_observation_v5_ptr,
            self._lib.trainv3_worker_observation_v5_len,
            (row_count, OBS_V5_DIM),
        )
        terminal_obs_v1 = self._optional_float_array(
            self._lib.trainv3_worker_terminal_observation_v1_ptr,
            self._lib.trainv3_worker_terminal_observation_v1_len,
            (row_count, OBS_V1_DIM),
            optional=self.observation_mode == "v5_only" or self.terminal_observation_mode == "none",
        )
        terminal_obs_v5 = self._optional_float_array(
            self._lib.trainv3_worker_terminal_observation_v5_ptr,
            self._lib.trainv3_worker_terminal_observation_v5_len,
            (row_count, OBS_V5_DIM),
            optional=self.terminal_observation_mode == "none",
        )
        mask = self._optional_float_array(
            self._lib.trainv3_worker_action_mask_ptr,
            self._lib.trainv3_worker_action_mask_len,
            (row_count, MAX_CANDIDATE_ACTIONS),
            optional=self.action_mask_mode == "legal_only",
        )
        feature_shape = (row_count, MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM)
        if self.action_features_mode == "legal_only":
            features = None
        elif self.action_features_dtype == "float16":
            features = self._float16_array(
                self._lib.trainv3_worker_action_features_f16_ptr,
                self._lib.trainv3_worker_action_features_f16_len,
                feature_shape,
            )
        else:
            features = self._float_array(
                self._lib.trainv3_worker_action_features_ptr,
                self._lib.trainv3_worker_action_features_len,
                feature_shape,
            )
        legal_counts = self._usize_array(
            self._lib.trainv3_worker_legal_action_counts_ptr,
            self._lib.trainv3_worker_legal_action_counts_len,
            (row_count,),
        )
        legal_offsets = self._usize_array(
            self._lib.trainv3_worker_legal_action_offsets_ptr,
            self._lib.trainv3_worker_legal_action_offsets_len,
            (row_count,),
        )
        legal_total = int(legal_counts.sum())
        legal_ids = self._usize_array(
            self._lib.trainv3_worker_legal_action_ids_ptr,
            self._lib.trainv3_worker_legal_action_ids_len,
            (legal_total,),
        )
        legal_feature_shape = (legal_total, ACTION_FEATURE_DIM)
        if self.action_features_dtype == "float16":
            legal_features = self._float16_array(
                self._lib.trainv3_worker_legal_action_features_f16_ptr,
                self._lib.trainv3_worker_legal_action_features_f16_len,
                legal_feature_shape,
            )
        else:
            legal_features = self._float_array(
                self._lib.trainv3_worker_legal_action_features_ptr,
                self._lib.trainv3_worker_legal_action_features_len,
                legal_feature_shape,
            )
        selected_local_indices = self._optional_i32_array(
            self._lib.trainv3_worker_selected_local_indices_ptr,
            self._lib.trainv3_worker_selected_local_indices_len,
            (row_count,),
        )
        rewards = self._float_array(
            self._lib.trainv3_worker_rewards_ptr,
            self._lib.trainv3_worker_rewards_len,
            (row_count,),
        )
        diagnostics_optional = self.diagnostic_mode == "none"
        episode_returns = self._optional_float_array(
            self._lib.trainv3_worker_episode_returns_ptr,
            self._lib.trainv3_worker_episode_returns_len,
            (row_count,),
            optional=diagnostics_optional,
        )
        episode_lengths = self._optional_usize_array(
            self._lib.trainv3_worker_episode_lengths_ptr,
            self._lib.trainv3_worker_episode_lengths_len,
            (row_count,),
            optional=diagnostics_optional,
        )
        terminated = self._u8_array(
            self._lib.trainv3_worker_terminated_ptr,
            self._lib.trainv3_worker_terminated_len,
            (row_count,),
        )
        terminated = _bool_view_from_u8(terminated)
        reset_flags_u8 = self._optional_u8_array(
            self._lib.trainv3_worker_reset_flags_ptr,
            self._lib.trainv3_worker_reset_flags_len,
            (row_count,),
            optional=diagnostics_optional,
        )
        reset_flags = None if reset_flags_u8 is None else _bool_view_from_u8(reset_flags_u8)
        terminal_obs_valid_u8 = self._optional_u8_array(
            self._lib.trainv3_worker_terminal_observation_valid_ptr,
            self._lib.trainv3_worker_terminal_observation_valid_len,
            (row_count,),
            optional=diagnostics_optional,
        )
        terminal_obs_valid = (
            None
            if terminal_obs_valid_u8 is None
            else _bool_view_from_u8(terminal_obs_valid_u8)
        )

        result = {
            "observation_v1": obs_v1,
            "observation_v5": obs_v5,
            "terminal_observation_v1": terminal_obs_v1,
            "terminal_observation_v5": terminal_obs_v5,
            "terminal_observation_valid": terminal_obs_valid,
            "action_mask": mask,
            "action_features": features,
            "legal_action_counts": legal_counts,
            "legal_action_offsets": legal_offsets,
            "legal_action_ids": legal_ids,
            "legal_action_features": legal_features,
            "selected_local_indices": selected_local_indices,
            "rewards": rewards,
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "terminated": terminated,
            "reset_flags": reset_flags,
        }
        if copy:
            result = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in result.items()
            }
        return result

    def _reshape_rollout_arrays(self, raw: dict[str, np.ndarray], *, steps: int) -> dict[str, np.ndarray]:
        shaped: dict[str, np.ndarray] = {}
        for key, value in raw.items():
            if value is None:
                shaped[key] = value
            elif key in {"legal_action_ids", "legal_action_features"}:
                shaped[key] = value
            else:
                shaped[key] = value.reshape((steps, self.env_count, *value.shape[1:]))
        if "legal_action_offsets" not in shaped:
            counts = shaped["legal_action_counts"]
            flat_counts = counts.reshape(-1)
            flat_offsets = np.empty_like(flat_counts, dtype=np.uintp)
            if flat_counts.size:
                flat_offsets[0] = 0
                if flat_counts.size > 1:
                    flat_offsets[1:] = np.cumsum(flat_counts[:-1], dtype=np.uintp)
            shaped["legal_action_offsets"] = flat_offsets.reshape(counts.shape)
        return shaped

    def _nonnull_ptr(self):
        if not getattr(self, "_ptr", None):
            raise RuntimeError("RustBatchWorker is closed")
        return self._ptr

    def _float_array(self, ptr_fn, len_fn, shape):
        ptr = ptr_fn(self._nonnull_ptr())
        length = len_fn(self._nonnull_ptr())
        expected = int(np.prod(shape))
        if not ptr or length != expected:
            raise RuntimeError(f"unexpected Rust f32 buffer length {length}, expected {expected}")
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(length,))
        return arr.reshape(shape)

    def _optional_float_array(self, ptr_fn, len_fn, shape, *, optional: bool):
        ptr = ptr_fn(self._nonnull_ptr())
        length = len_fn(self._nonnull_ptr())
        if optional and length == 0:
            return None
        expected = int(np.prod(shape))
        if not ptr or length != expected:
            raise RuntimeError(f"unexpected Rust f32 buffer length {length}, expected {expected}")
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(length,))
        return arr.reshape(shape)

    def _u8_array(self, ptr_fn, len_fn, shape):
        ptr = ptr_fn(self._nonnull_ptr())
        length = len_fn(self._nonnull_ptr())
        expected = int(np.prod(shape))
        if not ptr or length != expected:
            raise RuntimeError(f"unexpected Rust u8 buffer length {length}, expected {expected}")
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), shape=(length,))
        return arr.reshape(shape)

    def _optional_u8_array(self, ptr_fn, len_fn, shape, *, optional: bool):
        ptr = ptr_fn(self._nonnull_ptr())
        length = len_fn(self._nonnull_ptr())
        if optional and length == 0:
            return None
        expected = int(np.prod(shape))
        if not ptr or length != expected:
            raise RuntimeError(f"unexpected Rust u8 buffer length {length}, expected {expected}")
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)), shape=(length,))
        return arr.reshape(shape)

    def _usize_array(self, ptr_fn, len_fn, shape):
        ptr = ptr_fn(self._nonnull_ptr())
        length = len_fn(self._nonnull_ptr())
        expected = int(np.prod(shape))
        if not ptr or length != expected:
            raise RuntimeError(f"unexpected Rust usize buffer length {length}, expected {expected}")
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_size_t)), shape=(length,))
        return arr.reshape(shape)

    def _optional_usize_array(self, ptr_fn, len_fn, shape, *, optional: bool):
        ptr = ptr_fn(self._nonnull_ptr())
        length = len_fn(self._nonnull_ptr())
        if optional and length == 0:
            return None
        expected = int(np.prod(shape))
        if not ptr or length != expected:
            raise RuntimeError(f"unexpected Rust usize buffer length {length}, expected {expected}")
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_size_t)), shape=(length,))
        return arr.reshape(shape)

    def _optional_i32_array(self, ptr_fn, len_fn, shape):
        ptr = ptr_fn(self._nonnull_ptr())
        length = len_fn(self._nonnull_ptr())
        if length == 0:
            return None
        expected = int(np.prod(shape))
        if not ptr or length != expected:
            raise RuntimeError(f"unexpected Rust i32 buffer length {length}, expected {expected}")
        arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32)), shape=(length,))
        return arr.reshape(shape)

    def _float16_array(self, ptr_fn, len_fn, shape):
        ptr = ptr_fn(self._nonnull_ptr())
        length = len_fn(self._nonnull_ptr())
        expected = int(np.prod(shape))
        if not ptr or length != expected:
            raise RuntimeError(f"unexpected Rust f16 buffer length {length}, expected {expected}")
        bits = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16)), shape=(length,))
        return bits.view(np.float16).reshape(shape)


def _load_library(path: Path) -> ctypes.CDLL:
    cache_key = Path(path).resolve()
    cached = _LIBRARY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lib = ctypes.CDLL(str(cache_key))
    if hasattr(lib, "trainv3_compute_gae"):
        lib.trainv3_compute_gae.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.trainv3_compute_gae.restype = ctypes.c_int
    if hasattr(lib, "trainv3_select_local_indices"):
        lib.trainv3_select_local_indices.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.trainv3_select_local_indices.restype = ctypes.c_int
    if hasattr(lib, "trainv3_prepare_ppo_batch"):
        lib.trainv3_prepare_ppo_batch.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.trainv3_prepare_ppo_batch.restype = ctypes.c_int
    if hasattr(lib, "trainv3_pad_legal_actions"):
        lib.trainv3_pad_legal_actions.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.trainv3_pad_legal_actions.restype = ctypes.c_int
    if hasattr(lib, "trainv3_pack_legal_action_rows"):
        lib.trainv3_pack_legal_action_rows.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        lib.trainv3_pack_legal_action_rows.restype = ctypes.c_int
    if hasattr(lib, "trainv3_padded_argmax_actions"):
        lib.trainv3_padded_argmax_actions.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.trainv3_padded_argmax_actions.restype = ctypes.c_int
    if hasattr(lib, "trainv3_compact_argmax_actions"):
        lib.trainv3_compact_argmax_actions.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.trainv3_compact_argmax_actions.restype = ctypes.c_int
    if hasattr(lib, "trainv3_dense_argmax_actions"):
        lib.trainv3_dense_argmax_actions.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.trainv3_dense_argmax_actions.restype = ctypes.c_int
    if hasattr(lib, "trainv3_repeat_row_indices"):
        lib.trainv3_repeat_row_indices.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
        ]
        lib.trainv3_repeat_row_indices.restype = ctypes.c_int
    lib.trainv3_worker_from_trace_json.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.trainv3_worker_from_trace_json.restype = ctypes.c_void_p
    lib.trainv3_worker_from_trace_json_with_options.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    lib.trainv3_worker_from_trace_json_with_options.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_with_options_v2"):
        lib.trainv3_worker_from_trace_json_with_options_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_with_options_v2.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_with_options_v3"):
        lib.trainv3_worker_from_trace_json_with_options_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_with_options_v3.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_with_options_v4"):
        lib.trainv3_worker_from_trace_json_with_options_v4.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_with_options_v4.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_with_options_v5"):
        lib.trainv3_worker_from_trace_json_with_options_v5.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_with_options_v5.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_with_options_v6"):
        lib.trainv3_worker_from_trace_json_with_options_v6.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_with_options_v6.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v2"):
        lib.trainv3_worker_from_trace_json_pool_with_options_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_pool_with_options_v2.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v3"):
        lib.trainv3_worker_from_trace_json_pool_with_options_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_pool_with_options_v3.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v4"):
        lib.trainv3_worker_from_trace_json_pool_with_options_v4.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_pool_with_options_v4.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v5"):
        lib.trainv3_worker_from_trace_json_pool_with_options_v5.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_pool_with_options_v5.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v6"):
        lib.trainv3_worker_from_trace_json_pool_with_options_v6.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_pool_with_options_v6.restype = ctypes.c_void_p
    if hasattr(lib, "trainv3_worker_from_trace_json_pool_with_options_v7"):
        lib.trainv3_worker_from_trace_json_pool_with_options_v7.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.trainv3_worker_from_trace_json_pool_with_options_v7.restype = ctypes.c_void_p
    lib.trainv3_worker_free.argtypes = [ctypes.c_void_p]
    lib.trainv3_worker_free.restype = None
    lib.trainv3_worker_encode.argtypes = [ctypes.c_void_p]
    lib.trainv3_worker_encode.restype = ctypes.c_int
    lib.trainv3_worker_reset.argtypes = [ctypes.c_void_p]
    lib.trainv3_worker_reset.restype = ctypes.c_int
    if hasattr(lib, "trainv3_worker_use_chacha_rng"):
        lib.trainv3_worker_use_chacha_rng.argtypes = [ctypes.c_void_p]
        lib.trainv3_worker_use_chacha_rng.restype = ctypes.c_int
    lib.trainv3_worker_reset_indices.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.trainv3_worker_reset_indices.restype = ctypes.c_int
    lib.trainv3_worker_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
    lib.trainv3_worker_step.restype = ctypes.c_int
    lib.trainv3_worker_step_auto_reset.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.trainv3_worker_step_auto_reset.restype = ctypes.c_int
    if hasattr(lib, "trainv3_worker_step_mana_draw"):
        lib.trainv3_worker_step_mana_draw.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        lib.trainv3_worker_step_mana_draw.restype = ctypes.c_int
    if hasattr(lib, "trainv3_worker_current_actor_ids"):
        lib.trainv3_worker_current_actor_ids.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
        ]
        lib.trainv3_worker_current_actor_ids.restype = ctypes.c_int
    if hasattr(lib, "trainv3_worker_hero_hp"):
        lib.trainv3_worker_hero_hp.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
        ]
        lib.trainv3_worker_hero_hp.restype = ctypes.c_int
    if hasattr(lib, "trainv3_worker_select_rule_actions"):
        lib.trainv3_worker_select_rule_actions.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
        ]
        lib.trainv3_worker_select_rule_actions.restype = ctypes.c_int
    if hasattr(lib, "trainv3_worker_advance_rule_until_actor"):
        lib.trainv3_worker_advance_rule_until_actor.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
        ]
        lib.trainv3_worker_advance_rule_until_actor.restype = ctypes.c_int
    if hasattr(lib, "trainv3_worker_rollout_action_tape"):
        lib.trainv3_worker_rollout_action_tape.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint8,
        ]
        lib.trainv3_worker_rollout_action_tape.restype = ctypes.c_int
    if hasattr(lib, "trainv3_worker_rollout_action_tape_pre_step"):
        lib.trainv3_worker_rollout_action_tape_pre_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint8,
        ]
        lib.trainv3_worker_rollout_action_tape_pre_step.restype = ctypes.c_int

    for name in [
        "trainv3_worker_env_count",
        "trainv3_worker_observation_v1_len",
        "trainv3_worker_observation_v5_len",
        "trainv3_worker_terminal_observation_v1_len",
        "trainv3_worker_terminal_observation_v5_len",
        "trainv3_worker_action_mask_len",
        "trainv3_worker_action_features_len",
        "trainv3_worker_action_features_f16_len",
        "trainv3_worker_legal_action_counts_len",
        "trainv3_worker_legal_action_offsets_len",
        "trainv3_worker_legal_action_ids_len",
        "trainv3_worker_legal_action_features_len",
        "trainv3_worker_legal_action_features_f16_len",
        "trainv3_worker_selected_local_indices_len",
        "trainv3_worker_rewards_len",
        "trainv3_worker_episode_returns_len",
        "trainv3_worker_episode_lengths_len",
        "trainv3_worker_terminated_len",
        "trainv3_worker_reset_flags_len",
        "trainv3_worker_terminal_observation_valid_len",
        "trainv3_worker_truncated_len",
        "trainv3_worker_mana_draw_legal_len",
    ]:
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_size_t

    for name in [
        "trainv3_worker_observation_v1_ptr",
        "trainv3_worker_observation_v5_ptr",
        "trainv3_worker_terminal_observation_v1_ptr",
        "trainv3_worker_terminal_observation_v5_ptr",
        "trainv3_worker_action_mask_ptr",
        "trainv3_worker_action_features_ptr",
        "trainv3_worker_action_features_f16_ptr",
        "trainv3_worker_legal_action_counts_ptr",
        "trainv3_worker_legal_action_offsets_ptr",
        "trainv3_worker_legal_action_ids_ptr",
        "trainv3_worker_legal_action_features_ptr",
        "trainv3_worker_legal_action_features_f16_ptr",
        "trainv3_worker_selected_local_indices_ptr",
        "trainv3_worker_rewards_ptr",
        "trainv3_worker_episode_returns_ptr",
        "trainv3_worker_episode_lengths_ptr",
        "trainv3_worker_terminated_ptr",
        "trainv3_worker_reset_flags_ptr",
        "trainv3_worker_terminal_observation_valid_ptr",
        "trainv3_worker_truncated_ptr",
        "trainv3_worker_mana_draw_legal_ptr",
    ]:
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_void_p

    _LIBRARY_CACHE[cache_key] = lib
    return lib


def _normalize_action_features_dtype(value: str) -> str:
    if value not in {"float32", "float16"}:
        raise ValueError("action_features_dtype must be float32 or float16")
    return value


def _normalize_action_features_mode(value: str) -> str:
    if value == "dense":
        value = "dense_and_legal"
    if value not in {"dense_and_legal", "legal_only"}:
        raise ValueError("action_features_mode must be dense_and_legal or legal_only")
    return value


def _normalize_reset_pool_mode(value: str) -> str:
    if value not in {"fixed", "cycle"}:
        raise ValueError("reset_pool_mode must be fixed or cycle")
    return value


def _normalize_observation_mode(value: str) -> str:
    if value not in {"v1_and_v5", "v5_only"}:
        raise ValueError("observation_mode must be v1_and_v5 or v5_only")
    return value


def _normalize_action_mask_mode(value: str) -> str:
    if value not in {"dense", "legal_only"}:
        raise ValueError("action_mask_mode must be dense or legal_only")
    return value


def _normalize_terminal_observation_mode(value: str) -> str:
    if value not in {"full", "none"}:
        raise ValueError("terminal_observation_mode must be full or none")
    return value


def _normalize_diagnostic_mode(value: str) -> str:
    if value not in {"full", "none"}:
        raise ValueError("diagnostic_mode must be full or none")
    return value


__all__ = [
    "RustBatchWorker",
    "RustCompactArgmaxActions",
    "RustDenseArgmaxActions",
    "RustPaddedArgmaxActions",
    "RustPaddedLegalActions",
    "RustPackedLegalRows",
    "RustPreparedPPOBatch",
    "compute_rust_compact_argmax_actions",
    "compute_rust_dense_argmax_actions",
    "compute_rust_gae_returns",
    "compute_rust_pad_legal_actions",
    "compute_rust_pack_legal_action_rows",
    "compute_rust_padded_argmax_actions",
    "compute_rust_prepare_ppo_batch",
    "compute_rust_repeat_row_indices",
    "compute_rust_selected_local_indices",
    "default_library_candidates",
    "resolve_library_path",
]
