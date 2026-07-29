"""V5 ONNX export-parity + integrity tests (Block E1 / E2 -- E-E3, E-E14,
E-E15, E-E16, E-E17, SPEC :174, :177).

This is the V5 companion to the V4-frozen ``tests/test_train_v2_onnx_export.py``
(V4 LOGIT_TOL=1e-3 at :237 is NOT edited). The tightened ``LOGIT_TOL_V5=1e-4``
(E-E3 measure-then-pin) lives HERE, in the NEW V5 test, not in the V4 file.

Scope
------

* ``test_v5_onnx_fallback_guard`` -- the E2 core deliverable. Tests the
  standalone pure-numpy guard ``_assert_v5_logits_finite_legal`` in ISOLATION
  (NaN/inf/no-legal-candidate -> RuntimeError; valid -> finite-legal argmax int)
  + the ``V5RlhfAdapter`` construction-time raise points (ValueError on missing
  inference+path; no-raise with a fake inference callable). Runs WITHOUT mlx.

* ``test_v5_onnxruntime_matches_mlx`` -- the V5 analogue of
  ``test_train_v2_onnx_export.py:143-176,209-291``: build a tiny V5 ckpt, export
  via E1, drive MLX (Metal) + ONNX-runtime (CPU) over N steps; assert logits /
  value / mana_draw_logit numeric fidelity + behavioral argmax + mana_draw
  selection agreement. SCOPE NOTE: the rlhf_env ``V5RlhfAdapter`` DISCARDS
  ``_mana_draw_logit`` (v5_rlhf_adapter.py:201, mana_draw-BLIND), so the deploy
  adapter is NOT validated here -- only the head + the standalone
  ``select_includes_mana_draw`` are. The prod ``_get_action_v5`` path (E5)
  exercises the 3rd output in deploy.

* ``test_v5_metal_vs_cpu_drift_probe`` (E-E16) -- measures the actual V5
  Metal-vs-CPU drift on the tiny ckpt so ``LOGIT_TOL_V5`` can be pinned
  data-driven. PRINTS the measured drift.

* ``test_v5_warm_start_faithful_layer_equality`` + ``test_v5_base_encoder_path_is_faithful``
  + ``test_v5_base_1456_faithful_layer_bit_close`` + ``test_v5_601_logits_intentionally_diverge``
  (E-E14, HONOR Q3 PARTIAL) -- RE-ASSERT the warm-start gate on the SHIPPED
  artifact (save warm-started V5 -> export to ONNX -> load torch mirror ->
  re-assert faithful-layer equality + base-1456 path identity + 601 intentional
  divergence). Mirrors ``tests/test_train_v2_warm_start_v5.py:122-158,197-239,
  241-268``. The V4-Max npz is READ-ONLY (skip-gated if gitignored/absent).

* ``test_v5_sidecar_codec_sync_invariant`` (SPEC :177) -- asserts the shipped
  V5 sidecar ``card_shape_version`` == ``ai.train_v2.v5_card_shape_v1.CARD_SHAPE_VERSION``
  AND that ``train_v3.obs_v5`` imports ``encode_card_shape_v5`` from the SAME
  module (the codec-sync invariant). Runs with torch+onnx (no mlx needed). Does
  NOT assert against ``rlhf_env.components.v5_trace.CARD_SHAPE_VERSION`` -- that
  is a DIFFERENT trace-recorder-only constant ('classic_card_shape_v1').

FROZEN-CLASSIC GUARD: ``tests/test_train_v2_onnx_export.py`` /
``tests/test_train_v2_warm_start_v5.py`` / ``v5_rlhf_adapter.py`` / any V4-frozen
file are NOT edited. ``v5_inference_guard.py`` is the ONLY new non-test module.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path bootstrap -- insert the worktree root (so ai.* + rlhf_env.* resolve)
# + TrainV3.5/python (so train_v3.* resolves) when run via `python -m pytest`
# from the worktree root. Mirrors the Block D test bootstrap pattern
# (test_c_to_d_handoff.py:29-31 inserts the train_v3 parent).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_WORKTREE_ROOT = _HERE.parents[4]  # .../glm-TrainV3.5Prep
_TV3_PY = _WORKTREE_ROOT / "TrainV3.5" / "python"
for _p in (str(_WORKTREE_ROOT), str(_TV3_PY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Module-level constants (E-E3, E-E15).
# ---------------------------------------------------------------------------
# Tightened V5 tolerance (E-E3 = 1e-4 measure-then-pin). The V4 LOGIT_TOL=1e-3
# at tests/test_train_v2_onnx_export.py:237 is NOT edited (frozen-classic).
LOGIT_TOL_V5 = 1e-4
# Mana-draw head tolerance (E-E15 default = LOGIT_TOL_V5; the measure-then-pin
# probe may loosen it to 5e-4 if the fresh head drifts more -- start at 1e-4).
MANA_DRAW_LOGIT_TOL_V5 = 1e-4

N_STEPS = 5

# The canonical V4-Max checkpoint (m4_balanced run, update 1190). Gitignored in
# worktrees; reachable in the main repo checkout via the absolute path below.
# Mirror of tests/test_train_v2_warm_start_v5.py:31-34.
DEFAULT_V4_MAX_NPZ = (
    "/Users/laveqox/Documents/ExtraArenaRaS/ai/train_v2/runs/"
    "m4_balanced_from_0950_20260522_144431/checkpoints/update_1190.npz"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_torch_v5_as_npz(torch_model, path: str, metadata: dict | None = None) -> None:
    """Save a ``TorchV5ActionConditionedPolicy`` to an MLX-style .npz (no mlx).

    Writes the torch params under the MLX npz key scheme
    (``base_encoder.layers.0.weight`` etc.) so ``export_v5_checkpoint_to_onnx``
    and ``load_v5_torch_from_mlx_checkpoint`` can read it back. This is the
    mlx-independent ckpt builder for the sidecar test (which must NOT need mlx).
    """
    import time

    from train_v3.export_onnx_v5 import V5_WEIGHT_MAP, _resolve_torch_param

    npz: dict[str, np.ndarray] = {}
    for mlx_key, torch_key in V5_WEIGHT_MAP:
        mod = _resolve_torch_param(torch_model, torch_key)
        if hasattr(mod, "weight") and mod.weight is not None:
            npz[f"{mlx_key}.weight"] = (
                mod.weight.detach().cpu().numpy().astype(np.float32)
            )
        if hasattr(mod, "bias") and mod.bias is not None:
            npz[f"{mlx_key}.bias"] = (
                mod.bias.detach().cpu().numpy().astype(np.float32)
            )
    meta = (metadata or {}).copy()
    meta.setdefault("model_version", "v5_split_encoder_mlx_v1")
    meta.setdefault("obs_dim", 7128)
    meta.setdefault("action_feature_dim", 171)
    meta.setdefault("max_candidate_actions", 601)
    meta.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    npz["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )
    np.savez(str(path), **npz)


def _build_tiny_torch_v5():
    """Build a tiny TorchV5ActionConditionedPolicy for the sidecar test."""
    import torch

    from train_v3.export_onnx_v5 import TorchV5ActionConditionedPolicy

    torch.manual_seed(42)
    m = TorchV5ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
    m.eval()
    return m


def _build_tiny_v5_npz_mlx(path: str) -> str:
    """Build a tiny V5 .npz ckpt via an MLX V5 policy + save_checkpoint.

    Used by the parity + drift-probe tests (both need mlx anyway). Seeds MLX
    before construction so the fresh-init weights are deterministic.
    """
    import mlx.core as mx

    from ai.train_v2.model_mlx import save_checkpoint
    from train_v3.v5_policy import V5ActionConditionedPolicy

    mx.random.seed(42)
    policy = V5ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
    mx.eval(policy.parameters())
    save_checkpoint(
        path,
        policy,
        metadata={
            "obs_dim": 7128,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "config": {"hidden_dim": 32, "action_hidden_dim": 16},
        },
    )
    return path


def _masked_top2_margin(logits: np.ndarray) -> tuple[int, float]:
    """Return (argmax, top-2 margin) over a [601] logit vector (all-legal)."""
    arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 601:
        return -1, float("inf")
    order = np.argsort(arr)[::-1]
    top = int(order[0])
    second = float(arr[order[1]]) if arr.shape[0] > 1 else float("-inf")
    return top, float(arr[order[0]] - second)


def _resolve_v4_max_npz_path_or_skip() -> str:
    """Resolve the V4-Max npz; skip if absent (gitignored in worktrees)."""
    from train_v3.warm_start_v5 import resolve_v4_max_npz_path

    env = os.environ.get("V4_MAX_NPZ_PATH")
    if not env and Path(DEFAULT_V4_MAX_NPZ).is_file():
        os.environ["V4_MAX_NPZ_PATH"] = DEFAULT_V4_MAX_NPZ
    try:
        p = resolve_v4_max_npz_path()
    except RuntimeError as exc:
        pytest.skip(f"V4-Max npz not provisioned (gitignored, absent): {exc}")
    if not p.is_file():
        pytest.skip(f"V4-Max npz resolved to non-existent path: {p}")
    return str(p)


# ============================================================================
# (a) V5 ONNX fallback guard (SPEC :174) -- the E2 core deliverable
# ============================================================================
class TestV5OnnxFallbackGuard:
    """The V5 deploy path CANNOT silently degrade to a rule-based policy.

    E2 tests the standalone pure-numpy guard ``_assert_v5_logits_finite_legal``
    in ISOLATION (E2-green does NOT depend on E5) + the ``V5RlhfAdapter``
    construction-time raise points WITHOUT triggering a real onnxruntime load.
    Runs WITHOUT mlx.
    """

    def test_guard_nan_logits_raises_runtime_error(self):
        from train_v3.v5_inference_guard import _assert_v5_logits_finite_legal

        logits = np.full(601, np.nan, dtype=np.float32)
        legal_mask = np.ones(601, dtype=bool)
        with pytest.raises(RuntimeError, match="non-finite"):
            _assert_v5_logits_finite_legal(logits, legal_mask)

    def test_guard_all_inf_logits_raises_runtime_error(self):
        from train_v3.v5_inference_guard import _assert_v5_logits_finite_legal

        logits = np.full(601, np.inf, dtype=np.float32)
        legal_mask = np.ones(601, dtype=bool)
        with pytest.raises(RuntimeError, match="non-finite"):
            _assert_v5_logits_finite_legal(logits, legal_mask)

    def test_guard_all_false_legal_mask_raises_runtime_error(self):
        from train_v3.v5_inference_guard import _assert_v5_logits_finite_legal

        logits = np.arange(601, dtype=np.float32)
        legal_mask = np.zeros(601, dtype=bool)
        with pytest.raises(RuntimeError, match="no finite masked-in candidate"):
            _assert_v5_logits_finite_legal(logits, legal_mask)

    def test_guard_valid_logits_returns_finite_legal_argmax(self):
        from train_v3.v5_inference_guard import _assert_v5_logits_finite_legal

        # Build logits where index 300 is the max among legal candidates.
        logits = np.linspace(-1.0, 1.0, 601, dtype=np.float32)
        logits[300] = 5.0  # clear winner
        logits[100] = 2.0  # second-best
        # Mask out index 300 -- the argmax must fall to the next legal max (100).
        legal_mask = np.ones(601, dtype=bool)
        legal_mask[300] = False
        chosen = _assert_v5_logits_finite_legal(logits, legal_mask)
        assert isinstance(chosen, int)
        assert chosen == 100, f"expected masked argmax 100, got {chosen}"

    def test_guard_2d_1x601_input_handled(self):
        from train_v3.v5_inference_guard import _assert_v5_logits_finite_legal

        logits = np.zeros((1, 601), dtype=np.float32)
        logits[0, 42] = 3.0
        legal_mask = np.ones(601, dtype=bool)
        chosen = _assert_v5_logits_finite_legal(logits, legal_mask)
        assert chosen == 42

    def test_guard_unsupported_shape_raises_value_error(self):
        from train_v3.v5_inference_guard import _assert_v5_logits_finite_legal

        logits = np.zeros((2, 601), dtype=np.float32)
        legal_mask = np.ones(601, dtype=bool)
        with pytest.raises(ValueError, match=r"\[601\] or \[1, 601\]"):
            _assert_v5_logits_finite_legal(logits, legal_mask)

    def test_adapter_missing_inference_and_path_raises_value_error(self):
        """V5RlhfAdapter({}) with no inference and no path -> ValueError
        (v5_rlhf_adapter.py:97-101)."""
        from rlhf_env.components.v5_rlhf_adapter import V5RlhfAdapter

        with pytest.raises(ValueError, match="requires either an"):
            V5RlhfAdapter({})

    def test_adapter_with_fake_inference_does_not_raise(self):
        """V5RlhfAdapter with a fake inference callable -> no raise (the
        test-injection path). The fake returns a (logits, value, mana_draw_logit)
        tuple but is never called here (construction-time only)."""
        from rlhf_env.components.v5_rlhf_adapter import V5RlhfAdapter

        def fake_infer(obs, action_features):
            return (
                np.zeros(601, dtype=np.float32),
                0.0,
                0.0,
            )

        adapter = V5RlhfAdapter({}, inference=fake_infer)
        assert adapter.name == "v5-deploy"
        assert adapter._inference is fake_infer

    def test_adapter_onnxruntime_import_failure_raise_point_structural(self):
        """The onnxruntime-import-failure path (:105-110) is env-dependent
        (ort IS installed here). Assert the raise point EXISTS by reading the
        source (a light structural check) rather than uninstalling ort."""
        import inspect

        from rlhf_env.components.v5_rlhf_adapter import V5RlhfAdapter

        src = inspect.getsource(V5RlhfAdapter._build_onnx_inference)
        assert "import onnxruntime" in src
        assert "RuntimeError" in src
        assert "onnxruntime" in src.lower()

    def test_adapter_train_v3_import_failure_raise_point_structural(self):
        """Unexpected encoder/import failures use the stable policy envelope."""
        import inspect

        from rlhf_env.components.v5_rlhf_adapter import V5RlhfAdapter

        src = inspect.getsource(V5RlhfAdapter.select_action)
        assert "from train_v3" in src
        assert "RuntimeError" in src
        assert "v5_policy_failure_error" in src
        assert '"unexpected_failure"' in src


# ============================================================================
# (b) V5 ONNX-runtime vs MLX parity (E-E3)
# ============================================================================
class TestV5OnnxParity:
    """MLX (Metal) vs ONNX-runtime (CPU) numeric + behavioral parity."""

    @pytest.fixture
    def tiny_v5_onnx(self, tmp_path):
        """Build a tiny V5 ckpt (MLX), export to ONNX, return (onnx_path, npz_path)."""
        pytest.importorskip("mlx")
        pytest.importorskip("torch")
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        from train_v3.export_onnx_v5 import export_v5_checkpoint_to_onnx

        npz_path = str(tmp_path / "tiny_v5.npz")
        onnx_path = str(tmp_path / "tiny_v5.onnx")
        _build_tiny_v5_npz_mlx(npz_path)
        export_v5_checkpoint_to_onnx(npz_path, onnx_path, opset=17)
        return onnx_path, npz_path

    def test_v5_onnxruntime_matches_mlx(self, tiny_v5_onnx):
        """Drive MLX + ONNX over N steps; assert logits / value /
        mana_draw_logit fidelity + behavioral argmax + mana_draw selection
        agreement. SCOPE: the rlhf_env V5RlhfAdapter DISCARDS
        _mana_draw_logit (mana_draw-BLIND), so the deploy adapter is NOT
        validated here -- only the head + standalone select_includes_mana_draw."""
        import mlx.core as mx
        import onnxruntime as ort

        from ai.train_v2.model_mlx import load_checkpoint
        from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V5_DIM
        from train_v3.mana_draw_head_v5 import select_includes_mana_draw
        from train_v3.v5_policy import V5ActionConditionedPolicy

        onnx_path, npz_path = tiny_v5_onnx

        # MLX policy loaded from the SAME npz (same weights as the ONNX).
        mlx_model = V5ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(mlx_model.parameters())
        load_checkpoint(npz_path, mlx_model)

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        np.random.seed(0)
        for step in range(N_STEPS):
            obs_np = np.random.randn(1, OBS_V5_DIM).astype(np.float32) * 0.1
            af_np = (
                np.random.randn(1, 601, ACTION_FEATURE_DIM).astype(np.float32) * 0.1
            )

            # MLX forward -- mana_draw_legal=None returns the RAW 3-tuple.
            logits_mlx, value_mlx, mana_draw_mlx = mlx_model(
                mx.array(obs_np), mx.array(af_np), mana_draw_legal=None
            )
            mx.eval(logits_mlx, value_mlx, mana_draw_mlx)

            # ONNX forward.
            outs = sess.run(
                ["logits", "value", "mana_draw_logit"],
                {"observation": obs_np, "action_features": af_np},
            )
            logits_onnx = np.asarray(outs[0], dtype=np.float32)  # [1,601]
            value_onnx = np.asarray(outs[1], dtype=np.float32).reshape(-1)  # [1]
            mana_draw_onnx = np.asarray(outs[2], dtype=np.float32).reshape(-1)  # [1]

            logits_mlx_np = np.asarray(logits_mlx, dtype=np.float32)  # [1,601]
            value_mlx_np = np.asarray(value_mlx, dtype=np.float32).reshape(-1)  # [1]
            mana_draw_mlx_np = np.asarray(mana_draw_mlx, dtype=np.float32).reshape(-1)

            # (1) logits numeric fidelity.
            max_logit_diff = float(
                np.max(np.abs(logits_mlx_np - logits_onnx))
            )
            assert max_logit_diff < LOGIT_TOL_V5, (
                f"step {step}: max|logits_mlx - logits_onnx| = {max_logit_diff:.3e} "
                f">= LOGIT_TOL_V5={LOGIT_TOL_V5:.0e}"
            )

            # (2) value numeric fidelity (after squeeze).
            value_diff = float(np.abs(value_mlx_np[0] - value_onnx[0]))
            assert value_diff < LOGIT_TOL_V5, (
                f"step {step}: |value_mlx - value_onnx| = {value_diff:.3e} "
                f">= LOGIT_TOL_V5={LOGIT_TOL_V5:.0e}"
            )

            # (3) mana_draw_logit numeric fidelity (NEW 3rd-output axis).
            mana_draw_diff = float(
                np.abs(mana_draw_mlx_np[0] - mana_draw_onnx[0])
            )
            assert mana_draw_diff < MANA_DRAW_LOGIT_TOL_V5, (
                f"step {step}: |mana_draw_mlx - mana_draw_onnx| = "
                f"{mana_draw_diff:.3e} >= MANA_DRAW_LOGIT_TOL_V5="
                f"{MANA_DRAW_LOGIT_TOL_V5:.0e}"
            )

            # (4) behavioral argmax when the MLX top-2 margin > tol.
            a_mlx, margin_mlx = _masked_top2_margin(logits_mlx_np[0])
            a_onnx, _ = _masked_top2_margin(logits_onnx[0])
            if margin_mlx > LOGIT_TOL_V5 and a_mlx != a_onnx:
                raise AssertionError(
                    f"step {step}: clear winner (margin={margin_mlx:.3e}) "
                    f"but argmax diverges: MLX={a_mlx}, ONNX={a_onnx}"
                )

            # (5) mana_draw selection agreement when margin > tol. Use a fixed
            # mana_draw_legal=True (the random obs has no real engine state;
            # this validates the selection function in isolation).
            if margin_mlx > LOGIT_TOL_V5:
                best_mlx = float(logits_mlx_np[0, a_mlx])
                best_onnx = float(logits_onnx[0, a_onnx])
                sel_mlx = select_includes_mana_draw(
                    float(mana_draw_mlx_np[0]), best_mlx, True
                )
                sel_onnx = select_includes_mana_draw(
                    float(mana_draw_onnx[0]), best_onnx, True
                )
                assert sel_mlx == sel_onnx, (
                    f"step {step}: mana_draw selection diverges: "
                    f"MLX={sel_mlx}, ONNX={sel_onnx}"
                )

    def test_v5_metal_vs_cpu_drift_probe(self, tiny_v5_onnx):
        """E-E16 measure-then-pin: emit the measured V5 Metal-vs-CPU drift
        so LOGIT_TOL_V5 can be pinned data-driven. PRINTS the measured drift."""
        import mlx.core as mx
        import onnxruntime as ort

        from ai.train_v2.model_mlx import load_checkpoint
        from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V5_DIM
        from train_v3.v5_policy import V5ActionConditionedPolicy

        onnx_path, npz_path = tiny_v5_onnx

        mlx_model = V5ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(mlx_model.parameters())
        load_checkpoint(npz_path, mlx_model)

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        np.random.seed(123)
        max_drift = 0.0
        max_value_drift = 0.0
        max_mana_draw_drift = 0.0
        for _ in range(N_STEPS):
            obs_np = np.random.randn(1, OBS_V5_DIM).astype(np.float32) * 0.1
            af_np = (
                np.random.randn(1, 601, ACTION_FEATURE_DIM).astype(np.float32) * 0.1
            )
            logits_mlx, value_mlx, mana_draw_mlx = mlx_model(
                mx.array(obs_np), mx.array(af_np), mana_draw_legal=None
            )
            mx.eval(logits_mlx, value_mlx, mana_draw_mlx)
            outs = sess.run(
                ["logits", "value", "mana_draw_logit"],
                {"observation": obs_np, "action_features": af_np},
            )
            drift = float(
                np.max(np.abs(np.asarray(logits_mlx) - np.asarray(outs[0])))
            )
            max_drift = max(max_drift, drift)
            v_drift = float(
                np.abs(
                    np.asarray(value_mlx).reshape(-1)[0]
                    - np.asarray(outs[1]).reshape(-1)[0]
                )
            )
            max_value_drift = max(max_value_drift, v_drift)
            md_drift = float(
                np.abs(
                    np.asarray(mana_draw_mlx).reshape(-1)[0]
                    - np.asarray(outs[2]).reshape(-1)[0]
                )
            )
            max_mana_draw_drift = max(max_mana_draw_drift, md_drift)

        # PRINT the measured drift so the measure-then-pin probe can settle
        # LOGIT_TOL_V5 / MANA_DRAW_LOGIT_TOL_V5 data-driven.
        print(
            f"\n[E-E16 drift probe] max|logits_mlx - logits_onnx| = {max_drift:.3e} "
            f"| value drift = {max_value_drift:.3e} "
            f"| mana_draw drift = {max_mana_draw_drift:.3e} "
            f"(LOGIT_TOL_V5={LOGIT_TOL_V5:.0e}, MANA_DRAW_LOGIT_TOL_V5="
            f"{MANA_DRAW_LOGIT_TOL_V5:.0e})"
        )
        # The probe MUST measure a finite drift (sanity -- not asserting it
        # is below tol here; the parity test does that. This is the pin probe).
        assert np.isfinite(max_drift), f"non-finite drift: {max_drift}"


# ============================================================================
# (c) Warm-start faithful-layer equality on the SHIPPED artifact (E-E14)
# ============================================================================
class TestV5WarmStartShippedArtifact:
    """RE-ASSERT the Q3 PARTIAL warm-start gate on the SHIPPED/exported artifact.

    Build a V5 policy (MLX), warm-start via ``load_v4_max_into_v5``, save via
    ``save_checkpoint``, export to ONNX via E1, load the torch mirror via
    ``load_v5_torch_from_mlx_checkpoint`` on the saved npz, and re-assert on the
    EXPORTED torch mirror. HONORS Q3 PARTIAL (no 601-bit-close; the fresh
    ``state_fuser.layers.0`` + extra SiLU break the trunk). Mirrors
    ``tests/test_train_v2_warm_start_v5.py:122-158,197-239,241-268``.

    ALL tests skip-gate on mlx absent AND V4-Max npz absent. The V4-Max npz is
    READ-ONLY (skip-gated if gitignored/absent). NO edit to
    ``tests/test_train_v2_warm_start_v5.py`` (V4-frozen).
    """

    FAITHFUL_PAIRS = [
        ("base_encoder.layers.0.weight", "state_encoder.layers.0.weight"),
        ("base_encoder.layers.0.bias", "state_encoder.layers.0.bias"),
        ("action_encoder.weight", "action_encoder.weight"),
        ("action_encoder.bias", "action_encoder.bias"),
    ]

    @pytest.fixture
    def shipped_artifact(self, tmp_path):
        """Build a warm-started V5 (MLX), save, export to ONNX, load the torch
        mirror. Returns (torch_v5, v4_torch, v4_max_npz, npz_path, onnx_path)."""
        pytest.importorskip("mlx")
        pytest.importorskip("torch")
        pytest.importorskip("onnx")

        import mlx.core as mx

        from ai.train_v2.export_onnx import load_torch_from_mlx_checkpoint as v4_load_torch
        from ai.train_v2.model_mlx import save_checkpoint
        from train_v3.export_onnx_v5 import (
            export_v5_checkpoint_to_onnx,
            load_v5_torch_from_mlx_checkpoint,
        )
        from train_v3.v5_policy import V5ActionConditionedPolicy
        from train_v3.warm_start_v5 import load_v4_max_into_v5

        v4_max_npz = _resolve_v4_max_npz_path_or_skip()

        # Build a V5 policy (default dims -- matches the V4-Max shapes the
        # warm-start expects) and warm-start it.
        mx.random.seed(42)
        v5 = V5ActionConditionedPolicy()  # default hidden_dim=256
        load_v4_max_into_v5(v5, npz_path=v4_max_npz)

        # Save the warm-started V5 to an npz (SHIPPED artifact).
        npz_path = str(tmp_path / "warm_v5.npz")
        save_checkpoint(
            npz_path,
            v5,
            metadata={
                "obs_dim": 7128,
                "action_feature_dim": 171,
                "max_candidate_actions": 601,
                "config": {"hidden_dim": 256, "action_hidden_dim": 128},
            },
        )

        # Export to ONNX via E1 (SHIPPED artifact).
        onnx_path = str(tmp_path / "warm_v5.onnx")
        export_v5_checkpoint_to_onnx(npz_path, onnx_path, opset=17)

        # Load the torch mirror from the saved npz (the EXPORTED torch mirror).
        torch_v5, _ = load_v5_torch_from_mlx_checkpoint(npz_path)
        torch_v5.eval()

        # V4 torch mirror from the V4-Max npz (the base-1456 identity source).
        v4_torch, _ = v4_load_torch(v4_max_npz)
        v4_torch.eval()

        return torch_v5, v4_torch, v4_max_npz, npz_path, onnx_path

    @pytest.mark.parametrize("v5_name,v4_name", FAITHFUL_PAIRS)
    def test_v5_warm_start_faithful_layer_equality(
        self, shipped_artifact, v5_name, v4_name
    ):
        """E-E14: on the EXPORTED torch mirror, base_encoder.layers.0 +
        action_encoder == V4-Max state_encoder.layers.0 + action_encoder
        (assert_array_equal + allclose(atol=1e-7), mirroring :145-158)."""
        import torch

        torch_v5, _, v4_max_npz, _, _ = shipped_artifact

        # Resolve the V5 torch param by bracket index (export_onnx_v5 uses
        # base_encoder[0] / action_encoder -- the torch nn.Sequential param
        # names are base_encoder.0.weight).
        v5_state = dict(torch_v5.state_dict())
        # Map MLX key -> torch state_dict key (layers.N -> .N.).
        torch_key = v5_name.replace("layers.0", "0")
        assert torch_key in v5_state, f"V5 torch missing param {torch_key}"

        # V4-Max source weights (read directly from the READ-ONLY npz).
        v4_npz = dict(np.load(v4_max_npz, allow_pickle=True))
        assert v4_name in v4_npz, f"V4-Max npz missing param {v4_name}"

        a = v5_state[torch_key].detach().cpu().numpy()
        b = np.asarray(v4_npz[v4_name], dtype=np.float32)
        assert a.shape == b.shape, (
            f"{v5_name} shape {a.shape} != V4 {v4_name} shape {b.shape}"
        )
        # EXACT byte equality (the binding gate -- tight tolerance, mirror :152).
        np.testing.assert_array_equal(a, b, err_msg=(
            f"FAITHFUL layer {v5_name} != V4 {v4_name} (must be EXACT on the "
            f"EXPORTED torch mirror)"
        ))
        # Redundant belt-and-braces (mirror :156).
        assert np.allclose(a, b, atol=1e-7, rtol=0.0), (
            f"FAITHFUL layer {v5_name} not allclose to V4 {v4_name} within 1e-7 "
            f"on the EXPORTED torch mirror"
        )

    def test_v5_base_encoder_path_is_faithful(self, shipped_artifact):
        """Feed a frozen base-1456 input through the torch mirror
        base_encoder[0] and assert it matches the V4 torch mirror
        state_encoder[0] output (the base-1456 identity, mirror :241-268)."""
        import torch

        torch_v5, v4_torch, _, _, _ = shipped_artifact

        np.random.seed(7)
        base = torch.from_numpy(
            (np.random.randn(2, 1456).astype(np.float32) * 0.1)
        )
        with torch.no_grad():
            v5_base = torch_v5.base_encoder[0](base)
            v4_state = v4_torch.state_encoder[0](base)
        np.testing.assert_array_equal(
            v5_base.numpy(), v4_state.numpy(), err_msg=(
                "base_encoder[0] must produce IDENTICAL output to V4 "
                "state_encoder[0] for the same base-1456 input (faithful path "
                "on the EXPORTED torch mirror)"
            )
        )

    def test_v5_base_1456_faithful_layer_bit_close(self, shipped_artifact):
        """E-E14 BONUS: re-assert the faithful LAYER-output bit-close on the
        SHIPPED/exported artifact (the identity test re-asserted on the torch
        mirror, NOT the MLX policy)."""
        import torch

        torch_v5, v4_torch, _, _, _ = shipped_artifact

        np.random.seed(11)
        base = torch.from_numpy(
            (np.random.randn(3, 1456).astype(np.float32) * 0.05)
        )
        with torch.no_grad():
            v5_base = torch_v5.base_encoder[0](base)
            v4_state = v4_torch.state_encoder[0](base)
        np.testing.assert_array_equal(
            v5_base.numpy(), v4_state.numpy(), err_msg=(
                "base_encoder[0](base_1456) != V4 state_encoder[0](base_1456) "
                "on the EXPORTED torch mirror (bit-close bonus)"
            )
        )

    def test_v5_601_logits_intentionally_diverge(self, shipped_artifact):
        """HONOR Q3 PARTIAL: assert NOT allclose(v5_logits, v4_logits, atol=1e-3)
        AND max_diff > 0.05 (mirrors :197-239 -- do NOT assert 601-bit-close;
        the fresh state_fuser.layers.0 + extra SiLU break the trunk)."""
        import torch

        from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V1_DIM, OBS_V5_DIM

        torch_v5, v4_torch, _, _, _ = shipped_artifact

        np.random.seed(123)
        v5_obs_np = np.random.randn(1, OBS_V5_DIM).astype(np.float32) * 0.1
        v4_obs_np = v5_obs_np[:, :OBS_V1_DIM].copy()
        af_np = np.random.randn(1, 601, ACTION_FEATURE_DIM).astype(np.float32) * 0.1

        with torch.no_grad():
            v5_logits, _, _ = torch_v5(
                torch.from_numpy(v5_obs_np), torch.from_numpy(af_np)
            )
            v4_logits, _ = v4_torch(
                torch.from_numpy(v4_obs_np), torch.from_numpy(af_np)
            )
        v5l = v5_logits.numpy()
        v4l = v4_logits.numpy()
        assert v5l.shape == (1, 601) and v4l.shape == (1, 601)

        # The 601 logits MUST NOT match V4-Max within tolerance (Q3 PARTIAL).
        assert not np.allclose(v5l, v4l, atol=1e-3, rtol=1e-3), (
            "V5 warm-started logits MUST NOT match V4-Max logits (Q3 partial: "
            "the fresh state_fuser.layers.0 + extra SiLU break the trunk)."
        )
        max_diff = float(np.max(np.abs(v5l - v4l)))
        assert max_diff > 0.05, (
            f"logit divergence too small ({max_diff:.6f}) -- expected a "
            "meaningful difference from the architectural disconnect, not noise"
        )


# ============================================================================
# (d) Sidecar codec-sync invariant (SPEC :177)
# ============================================================================
class TestV5SidecarCodecSync:
    """The shipped V5 sidecar ``card_shape_version`` must equal
    ``ai.train_v2.v5_card_shape_v1.CARD_SHAPE_VERSION`` AND the obs encoder
    (``train_v3.obs_v5``) must import ``encode_card_shape_v5`` from the SAME
    module whose version is in the sidecar (the codec-sync invariant).

    Runs with torch+onnx -- no mlx needed. Does NOT assert against
    ``rlhf_env.components.v5_trace.CARD_SHAPE_VERSION`` -- that is a DIFFERENT
    trace-recorder-only constant ('classic_card_shape_v1') and would FAIL; the
    rlhf_env deploy path uses ``train_v3.obs_v5`` (same v5_card_shape_v1
    module), so the invariant is sidecar == v5_card_shape_v1.CARD_SHAPE_VERSION.
    """

    def test_v5_sidecar_codec_sync_invariant(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("onnx")

        from ai.train_v2.v5_card_shape_v1 import CARD_SHAPE_VERSION
        from train_v3.export_onnx_v5 import export_v5_checkpoint_to_onnx

        # Build a tiny V5 ckpt from the torch mirror (NO mlx needed for the
        # sidecar-only export -- the torch mirror's weights are written to an
        # MLX-style npz via _save_torch_v5_as_npz).
        torch_model = _build_tiny_torch_v5()
        npz_path = str(tmp_path / "sidecar_v5.npz")
        _save_torch_v5_as_npz(
            torch_model,
            npz_path,
            metadata={
                "obs_dim": 7128,
                "action_feature_dim": 171,
                "max_candidate_actions": 601,
                "config": {"hidden_dim": 32, "action_hidden_dim": 16},
            },
        )

        onnx_path = str(tmp_path / "sidecar_v5.onnx")
        export_v5_checkpoint_to_onnx(npz_path, onnx_path, opset=17)

        sidecar = json.loads(Path(onnx_path + ".json").read_text())
        assert sidecar["card_shape_version"] == CARD_SHAPE_VERSION, (
            f"sidecar card_shape_version {sidecar['card_shape_version']!r} != "
            f"v5_card_shape_v1.CARD_SHAPE_VERSION {CARD_SHAPE_VERSION!r}"
        )
        assert CARD_SHAPE_VERSION == "v5_card_shape_v1"

        # Structural codec-sync: train_v3.obs_v5 imports encode_card_shape_v5
        # from ai.train_v2.v5_card_shape_v1 (the SAME module whose version is in
        # the sidecar). Verify the import resolves + the symbol is callable.
        from train_v3 import obs_v5 as obs_v5_mod

        assert hasattr(obs_v5_mod, "encode_card_shape_v5"), (
            "train_v3.obs_v5 must re-export encode_card_shape_v5 (the codec-sync "
            "structural invariant)"
        )
        assert callable(obs_v5_mod.encode_card_shape_v5)
        # The obs encoder MUST import it from ai.train_v2.v5_card_shape_v1
        # (the same module whose CARD_SHAPE_VERSION is in the sidecar). Verify
        # by checking the source module of encode_card_shape_v5.
        import inspect

        src_mod = inspect.getmodule(obs_v5_mod.encode_card_shape_v5)
        assert src_mod is not None
        assert src_mod.__name__ == "ai.train_v2.v5_card_shape_v1", (
            f"encode_card_shape_v5 is defined in {src_mod.__name__}, not "
            "ai.train_v2.v5_card_shape_v1 (the codec-sync invariant is broken)"
        )
