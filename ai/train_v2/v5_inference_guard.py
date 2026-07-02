"""V5 ONNX fallback guard -- pure-numpy logits-finite / argmax-legal check.

SPEC design.md:174 -- the standalone ONNX fallback guard. The V5 deploy path
(``V5RlhfAdapter.select_action`` / the prod ``_get_action_v5`` path in
``ai/bot_brain.py``) must NEVER silently degrade to a rule-based policy when a
V5 ONNX loads but produces NaN/garbage logits. The current
``V5RlhfAdapter.select_action:206-208`` does a SILENT ``return 0`` on a no-legal
candidate; this guard replaces that silent path with a hard ``RuntimeError`` so
a malformed V5 ONNX is caught at the last line of defense, NOT masked as a
plausible-but-wrong action 0.

This module is the SOURCE-OF-TRUTH for the guard. It is PURE-NUMPY (only numpy
+ typing) -- NO mlx/torch/onnx/onnxruntime import and NO train_v3-internal
import -- so it is PROD-VENDORABLE by E5 (mirrors the ``mana_draw_head_v5``
vendoring pattern: a pure-Python/numpy module that the prod path can copy in
without dragging the MLX training stack). E2 tests it in ISOLATION (a fake
inference callable); E5 wires the tested function into ``V5RlhfAdapter.select_action``
and the prod ``_get_action_v5`` path.

Contract
---------

``_assert_v5_logits_finite_legal(logits, legal_mask) -> int``

  * ``logits`` -- a 1-D numpy array of shape ``[601]`` or a 2-D ``[1, 601]``
    (the per-step primary case). Other shapes raise ``ValueError``. The 2-D
    ``[1, 601]`` form is collapsed to row 0.

  * ``legal_mask`` -- a 1-D bool numpy array (shape ``[601]`` or broadcastable
    to the collapsed logits).

  * Returns the finite-legal argmax (``int``) so E5 can use the return value
    directly as a drop-in for the current silent ``return 0`` path.

  * Raises ``RuntimeError`` on:
      (1) any non-finite logit (NaN or +/-inf) -- a malformed V5 ONNX is the
          last-resort prod safety, NOT a silent rule-based fallback;
      (2) no finite masked-in candidate (illegal state -- every candidate is
          masked out, which is an engine/encoder bug, not a policy decision).

  * The guard NEVER silently returns 0 -- it either returns a real
    finite-legal argmax or raises ``RuntimeError``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["_assert_v5_logits_finite_legal"]


def _assert_v5_logits_finite_legal(logits: Any, legal_mask: Any) -> int:
    """Validate V5 ONNX logits + return the finite-legal argmax (or raise).

    See the module docstring for the full contract. Pure-numpy: no MLX/torch/
    onnxruntime import. Raises ``ValueError`` on an unsupported ``logits`` shape
    and ``RuntimeError`` on NaN/inf logits or no finite masked-in candidate.
    """
    arr = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(legal_mask, dtype=bool)

    # Collapse to a 1-D [601] vector. Support the per-step primary case [601]
    # and the batched [1, 601] form (row 0); reject any other shape.
    if arr.ndim == 1:
        flat = arr
    elif arr.ndim == 2 and arr.shape[0] == 1:
        flat = arr[0]
    else:
        raise ValueError(
            f"_assert_v5_logits_finite_legal expects logits of shape [601] or "
            f"[1, 601], got shape {tuple(arr.shape)}"
        )

    if flat.shape != (601,):
        raise ValueError(
            f"_assert_v5_logits_finite_legal expects 601 logits, got "
            f"flattened shape {tuple(flat.shape)}"
        )

    # (1) FINITE check -- NaN/inf logits from a malformed V5 ONNX are the
    # last-resort prod safety. NOT a silent rule-based fallback.
    if np.any(~np.isfinite(flat)):
        raise RuntimeError(
            "V5 ONNX fallback guard: non-finite logits detected (NaN or inf). "
            "A malformed V5 ONNX must NOT silently degrade to a rule-based "
            "policy -- the deploy path raises rather than masking the failure."
        )

    # (2) LEGAL-CANDIDATE check -- illegal state if every candidate is masked
    # out (engine/encoder bug, not a policy decision).
    masked = np.where(mask, flat, np.float32(-np.inf)).astype(np.float32)
    if not np.any(np.isfinite(masked)):
        raise RuntimeError(
            "V5 ONNX fallback guard: no finite masked-in candidate -- the "
            "legal_mask masks out all 601 candidates. This is an illegal state "
            "(engine/encoder bug), not a policy decision."
        )

    # (3) success -- the finite-legal argmax. E5 uses this directly as a
    # drop-in for the current silent return-0 path.
    return int(np.argmax(masked))