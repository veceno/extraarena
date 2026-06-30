"""V4-Max -> V5 partial warm-start loader (Block 0 component 4 / spec
§6.188 + §6.189, Q3 PARTIAL verdict).

Q3 PARTIAL VERDICT (BLOCK0_FOUNDATION_PLAN.md:12-16): the V5 split-encoder CANNOT
reproduce the V4-Max 1456->256->601 trunk. The fresh ``state_fuser.layers.0
(544,256)`` plus the extra SiLU between the base projection and the 2nd linear
breaks the V4 trunk (SiLU is nonlinear -> no identity init is possible).
Warm-start is therefore a PARTIAL name+shape remap, NOT a parity transfer:

  FAITHFUL (exact transfer -- reproduces V4 function fed from obs /
  action_features, since the V5 base_encoder.layers.0 consumes the SAME first
  1456 dims of the V5 obs as V4's state_encoder.layers.0, and the V5
  action_encoder consumes the SAME action_features as V4's action_encoder):
    V5 base_encoder.layers.0 (256,1456) <- V4 state_encoder.layers.0
    V5 action_encoder        (128,171) <- V4 action_encoder

  SHAPE-COMPAT-DISCONNECTED (loads by shape, but the fused-state INPUT differs
  so the 601 logits will NOT match V4-Max -- the V5 candidate_scorer /
  value_head / state_fuser.layers.2 read the FUSED state_emb produced by
  state_fuser.layers.0 over [base|global|private|history], NOT the V4
  state_encoder.layers.0 output):
    V5 state_fuser.layers.2 (256,256) <- V4 state_encoder.layers.2
    V5 candidate_scorer     (1,384)   <- V4 candidate_scorer
    V5 value_head           (1,256)   <- V4 value_head

  FRESH (no V4 counterpart -- leave default MLX init, do NOT touch):
    V5 global_encoder.layers.0   (32,32)
    V5 private_encoder.layers.0  (128,2400)   [2400 not 2112 -- component-1 grow]
    V5 history_encoder.layers.0 (128,3240)   [3240 not 2880 -- component-1 grow]
    V5 state_fuser.layers.0     (256,544)
    V5 mana_draw_head            (1,256)

The binding gate is FAITHFUL-LAYER EQUALITY (spec §6.188 full-forward-parity is
RELAXED per Q3): after load, V5 ``base_encoder.layers.0`` + ``action_encoder``
must byte-match the V4-Max source. The 601 logits will DIFFER (the
shape-compat-disconnected path feeds a different fused state), which is asserted
in the companion tests to prevent the false confidence that warm-start = parity.

FROZEN-CLASSIC GUARD: ``classic_*`` byte-frozen, untouched. The V4-Max npz +
ONNX are READ-ONLY -- the loader NEVER writes to them. Warm-start = "head
start", not parity (Q3).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# The canonical V4-Max checkpoint (m4_balanced run, update 1190). This relative
# path is searched against repo-root candidates (walk-up from this file + CWD)
# when no explicit path / env is provided. It is GITIGNORED in worktrees, so the
# walk-up must reach the main repo checkout to find it.
DEFAULT_NPZ_REL = (
    "ai/train_v2/runs/m4_balanced_from_0950_20260522_144431/checkpoints/"
    "update_1190.npz"
)

# ---------------------------------------------------------------------------
# Transfer map (Q3 PARTIAL). Layer-level -> param-level is derived below.
# ---------------------------------------------------------------------------
# Each entry: (v5_layer_base, v4_layer_base, category).
# Only FAITHFUL + SHAPE-COMPAT-DISCONNECTED layers are loaded from V4; FRESH
# layers are left at their default MLX init (untouched).
TRANSFER_MAP_LAYERS: list[tuple[str, str, str]] = [
    ("base_encoder.layers.0", "state_encoder.layers.0", "faithful"),
    ("action_encoder", "action_encoder", "faithful"),
    ("state_fuser.layers.2", "state_encoder.layers.2", "shape-compat-disconnected"),
    ("candidate_scorer", "candidate_scorer", "shape-compat-disconnected"),
    ("value_head", "value_head", "shape-compat-disconnected"),
]

# V5 layers with no V4 counterpart (left at default init). Listed explicitly so
# the report can document them and the tests can assert they are NOT overwritten.
FRESH_V5_LAYERS: list[str] = [
    "global_encoder.layers.0",
    "private_encoder.layers.0",
    "history_encoder.layers.0",
    "state_fuser.layers.0",
    "mana_draw_head",
]

# Expand the layer map to a param-level map: {v5_param_name: (v4_param_name,
# category)} for both .weight and .bias. This is what the loader walks against
# the flattened V5 param tree.
TRANSFER_MAP: dict[str, tuple[str, str]] = {}
for _v5_base, _v4_base, _cat in TRANSFER_MAP_LAYERS:
    for _suffix in (".weight", ".bias"):
        TRANSFER_MAP[_v5_base + _suffix] = (_v4_base + _suffix, _cat)

# The 10 V4 weight (non-opt, non-meta) keys expected in the npz. Used by the
# strict-false-no-silent-drops test to assert every V4 weight key is mapped.
V4_WEIGHT_KEYS_EXPECTED: list[str] = [
    "state_encoder.layers.0.weight",
    "state_encoder.layers.0.bias",
    "state_encoder.layers.2.weight",
    "state_encoder.layers.2.bias",
    "action_encoder.weight",
    "action_encoder.bias",
    "candidate_scorer.weight",
    "candidate_scorer.bias",
    "value_head.weight",
    "value_head.bias",
]

_VALID_CATEGORIES = ("faithful", "shape-compat-disconnected", "fresh")


def resolve_v4_max_npz_path(npz_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the V4-Max npz path.

    Resolution order:
      1. ``npz_path`` argument (if provided -- must exist, else RuntimeError).
      2. ``V4_MAX_NPZ_PATH`` environment variable (must exist, else RuntimeError).
      3. Default candidate search: walk up from this file's directory and from
         the CWD, checking for ``DEFAULT_NPZ_REL`` under each ancestor. The npz
         is gitignored in worktrees, so the walk-up must reach the main repo
         checkout to find it.

    Raises ``RuntimeError`` with a clear reason if no candidate exists.
    """
    if npz_path is not None:
        p = Path(npz_path).expanduser()
        if not p.is_file():
            raise RuntimeError(
                f"V4-Max npz not found at explicit path: {p} "
                "(set V4_MAX_NPZ_PATH env or pass a valid npz_path)"
            )
        return p.resolve()

    env = os.environ.get("V4_MAX_NPZ_PATH")
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise RuntimeError(
                f"V4-Max npz not found at V4_MAX_NPZ_PATH={env} "
                "(unset the env to fall back to the default candidate search)"
            )
        return p.resolve()

    # Default candidate search: walk up from this file's location AND from CWD.
    here = Path(__file__).resolve()
    cwd = Path.cwd()
    candidates: list[Path] = []
    for root in (here.parent, cwd):
        for ancestor in [root, *root.parents]:
            candidate = ancestor / DEFAULT_NPZ_REL
            if candidate not in candidates:
                candidates.append(candidate)
    for c in candidates:
        if c.is_file():
            return c.resolve()

    searched = "\n  ".join(str(c) for c in candidates)
    raise RuntimeError(
        "V4-Max npz not found. Set V4_MAX_NPZ_PATH env or pass npz_path to "
        "load_v4_max_into_v5. The npz is gitignored in worktrees and lives only "
        "in the main repo checkout. Searched candidates:\n  " + searched
    )


def _parse_npz_meta(meta_arr: np.ndarray) -> dict[str, Any]:
    """Decode the ``__meta__`` uint8 array stored by ``model_mlx.save_checkpoint``
    (a JSON byte buffer). Returns ``{}`` if absent or unparseable."""
    if meta_arr is None:
        return {}
    try:
        raw = meta_arr.tobytes().decode("utf-8")
        return json.loads(raw)
    except Exception:
        return {}


def load_v4_max_into_v5(
    policy,
    npz_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Warm-start a ``V5ActionConditionedPolicy`` from the V4-Max npz (Q3 PARTIAL).

    Loads the V4-Max checkpoint (``update_1190.npz``) into ``policy`` applying
    the Q3 transfer map:

      * FAITHFUL params are copied EXACTLY (V5 reproduces V4 function for the
        base-1456 + action-features subset).
      * SHAPE-COMPAT-DISCONNECTED params are copied by shape (inputs differ, so
        the 601 logits will NOT match V4-Max -- documented, asserted in tests).
      * FRESH params are left at their default MLX init (untouched).

    This is the strict=False equivalent: only mapped params are overwritten;
    every other V5 param keeps its construction-time init. The V4-Max npz is
    READ-ONLY (the loader never writes to it).

    Returns a report dict:
      ``transferred``  -- list of {v5_layer, v4_source, category, shape} (loaded)
      ``dropped``      -- list of {v5_layer, v4_source, reason} (failed to load;
                          empty on success)
      ``fresh``        -- list of V5 layer names left at default init
      ``v4_weight_keys``       -- V4 weight keys found in the npz
      ``v4_opt_keys_skipped``  -- V4 ``_opt_`` optimizer-state keys (not loaded
                          by design -- warm-start is a policy transfer, not an
                          optimizer restore)
      ``npz_path`` -- resolved npz path (str)
      ``npz_meta`` -- parsed ``__meta__`` dict
    """
    import mlx.core as mx
    import mlx.nn as nn

    resolved = resolve_v4_max_npz_path(npz_path)

    # allow_pickle=False is safe: the npz holds only numeric arrays (float32
    # weights/biases, uint64/float32 scalar Adam state, uint8 __meta__ bytes) --
    # no object arrays require pickle.
    with np.load(str(resolved), allow_pickle=False) as data:
        npz_keys = list(data.files)
        npz_arrays = {k: data[k] for k in npz_keys}

    # Partition V4 keys: weights vs optimizer state vs metadata.
    v4_weight_keys = [k for k in npz_keys if not k.startswith("_opt_") and k != "__meta__"]
    v4_opt_keys = [k for k in npz_keys if k.startswith("_opt_")]
    npz_meta = _parse_npz_meta(npz_arrays.get("__meta__"))

    # Flatten the V5 param tree (dotted-name -> mx.array), preserving order.
    flat = nn.utils.tree_flatten(policy.trainable_parameters())  # list[(name, val)]

    new_pairs: list[tuple[str, mx.array]] = []
    transferred: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []

    for name, val in flat:
        mapping = TRANSFER_MAP.get(name)
        if mapping is None:
            # FRESH: leave default init.
            new_pairs.append((name, val))
            continue
        v4_name, category = mapping
        if v4_name not in npz_arrays:
            dropped.append({
                "v5_layer": name,
                "v4_source": v4_name,
                "reason": "v4 key absent from npz",
            })
            new_pairs.append((name, val))
            continue
        v4_arr = npz_arrays[v4_name]
        if tuple(v4_arr.shape) != tuple(val.shape):
            dropped.append({
                "v5_layer": name,
                "v4_source": v4_name,
                "reason": f"shape mismatch v5={tuple(val.shape)} v4={tuple(v4_arr.shape)}",
            })
            new_pairs.append((name, val))
            continue
        new_pairs.append((name, mx.array(v4_arr.astype(np.float32))))
        transferred.append({
            "v5_layer": name,
            "v4_source": v4_name,
            "category": category,
            "shape": str(tuple(v4_arr.shape)),
        })

    # Rebuild the full param tree (mapped params swapped, fresh params kept) and
    # apply. This is strict=False semantics: only mapped params change.
    tree = nn.utils.tree_unflatten(new_pairs)
    policy.update(tree)
    mx.eval(policy.parameters())

    return {
        "transferred": transferred,
        "dropped": dropped,
        "fresh": list(FRESH_V5_LAYERS),
        "v4_weight_keys": v4_weight_keys,
        "v4_opt_keys_skipped": v4_opt_keys,
        "npz_path": str(resolved),
        "npz_meta": npz_meta,
    }


__all__ = [
    "DEFAULT_NPZ_REL",
    "FRESH_V5_LAYERS",
    "TRANSFER_MAP",
    "TRANSFER_MAP_LAYERS",
    "V4_WEIGHT_KEYS_EXPECTED",
    "load_v4_max_into_v5",
    "resolve_v4_max_npz_path",
]