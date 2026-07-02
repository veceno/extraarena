"""Tests for the V4-Max -> V5 partial warm-start loader (Block 0 component 4 /
spec §6.188 + §6.189, Q3 PARTIAL verdict).

SOURCE-VS-SOURCE ORACLE (Block -1 lesson): the warm-start is asserted against
the REAL V4-Max checkpoint and the REAL V4 ``model_mlx.ActionConditionedPolicy``,
NOT a self-referential fixture. The binding gate is FAITHFUL-LAYER EQUALITY
(``base_encoder.layers.0`` + ``action_encoder`` byte-match the V4 source after
load); spec §6.188 full-forward-pass-parity is RELAXED per Q3 (the 601 logits
DIFFER -- the fresh ``state_fuser.layers.0`` + extra SiLU breaks the trunk).

FROZEN-CLASSIC GUARD: ``classic_*`` byte-frozen, untouched. The V4-Max npz +
ONNX are READ-ONLY (the loader never writes them).

SKIP GATE: the binding skip is NPZ ABSENCE only (the npz is gitignored in
worktrees and lives only in the main repo checkout). MLX is importable in this
worktree (component 3 confirmed the Q3 "MLX not importable" note was wrong);
``pytest.importorskip("mlx")`` is a no-op here and only guards other envs.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# The canonical V4-Max checkpoint (m4_balanced run, update 1190). Gitignored in
# worktrees; reachable in the main repo checkout via the absolute path below.
# Tests that need it resolve via ``resolve_v4_max_npz_path`` (env ->
# candidate-search walk-up -> clear skip if absent).
DEFAULT_V4_MAX_NPZ = (
    "/Users/laveqox/Documents/ExtraArenaRaS/ai/train_v2/runs/"
    "m4_balanced_from_0950_20260522_144431/checkpoints/update_1190.npz"
)

# The V4 architecture (model_mlx.py:37-81) as key->shape, for the npz-key dump
# test. These are the 10 weight (non-opt, non-meta) params the loader maps.
EXPECTED_V4_WEIGHT_SHAPES = {
    "state_encoder.layers.0.weight": (256, 1456),
    "state_encoder.layers.0.bias": (256,),
    "state_encoder.layers.2.weight": (256, 256),
    "state_encoder.layers.2.bias": (256,),
    "action_encoder.weight": (128, 171),
    "action_encoder.bias": (128,),
    "candidate_scorer.weight": (1, 384),
    "candidate_scorer.bias": (1,),
    "value_head.weight": (1, 256),
    "value_head.bias": (1,),
}


@pytest.fixture(scope="module")
def v4_max_npz_path():
    """Resolve the V4-Max npz; skip the whole module if absent (gitignored,
    not provisioned). The skip is on NPZ ABSENCE only -- MLX is importable here."""
    from train_v3.warm_start_v5 import resolve_v4_max_npz_path

    # Prefer the documented absolute path / env so the test runs in this
    # worktree (the walk-up also reaches the main repo checkout, but the env
    # makes the dependency explicit per the component-4 task).
    env = os.environ.get("V4_MAX_NPZ_PATH")
    if not env and Path(DEFAULT_V4_MAX_NPZ).is_file():
        os.environ["V4_MAX_NPZ_PATH"] = DEFAULT_V4_MAX_NPZ
    try:
        p = resolve_v4_max_npz_path()
    except RuntimeError as exc:  # pragma: no cover - skip path
        pytest.skip(f"V4-Max npz not provisioned (gitignored, absent): {exc}")
    if not p.is_file():  # pragma: no cover - defensive
        pytest.skip(f"V4-Max npz resolved to non-existent path: {p}")
    return str(p)


@pytest.fixture()
def loaded_v5_and_v4(v4_max_npz_path):
    """Build a seeded V5 policy, warm-start it from the npz, and build the V4
    policy loaded from the SAME npz (canonical ``load_checkpoint``). Both are
    deterministic (seeded V5 fresh-init + V4-from-npz)."""
    mlx = pytest.importorskip("mlx")  # no-op here; guards other envs
    import mlx.core as mx
    import mlx.nn as nn

    from ai.train_v2.model_mlx import ActionConditionedPolicy as V4Policy
    from ai.train_v2.model_mlx import load_checkpoint
    from train_v3.v5_policy import V5ActionConditionedPolicy
    from train_v3.warm_start_v5 import load_v4_max_into_v5

    npz = v4_max_npz_path
    # Seed BEFORE constructing V5 so the fresh layers have deterministic init
    # (then load_v4_max_into_v5 overwrites the mapped params; fresh ones keep
    # this seeded init).
    mx.random.seed(42)
    v5 = V5ActionConditionedPolicy()
    report = load_v4_max_into_v5(v5, npz_path=npz)

    v4 = V4Policy()
    load_checkpoint(npz, v4)  # canonical V4 loader (allow_pickle=True, pops meta)

    return v5, v4, report


# ============================================================================
# (a) V4-Max npz present or skip
# ============================================================================
class TestV4MaxNpzPresentOrSkip:
    def test_v4_max_npz_present_or_skip(self, v4_max_npz_path):
        """The binding skip-gate: if the resolved npz path does not exist, skip
        with a clear reason. When present, assert it is a non-empty .npz file
        with the expected checkpoint name (documents the gate explicitly)."""
        p = Path(v4_max_npz_path)
        assert p.is_file(), f"resolved npz is not a file: {p}"
        assert p.suffix == ".npz"
        assert p.stat().st_size > 0, f"npz is empty: {p}"
        # The canonical V4-Max checkpoint name (m4_balanced run, update 1190).
        assert p.name == "update_1190.npz", (
            f"unexpected npz name {p.name} (expected update_1190.npz)"
        )


# ============================================================================
# (b) Faithful-layer equality -- THE BINDING GATE
# ============================================================================
class TestFaithfulLayerEquality:
    """Q3 FAITHFUL transfer: V5 ``base_encoder.layers.0`` + ``action_encoder``
    must EXACTLY equal the V4-Max source after load (these are the layers that
    reproduce V4 function fed from obs/action_features). This is the binding
    gate -- spec §6.188 full-forward-parity is RELAXED, but faithful-layer
    equality is NOT."""

    FAITHFUL_PAIRS = [
        ("base_encoder.layers.0.weight", "state_encoder.layers.0.weight"),
        ("base_encoder.layers.0.bias", "state_encoder.layers.0.bias"),
        ("action_encoder.weight", "action_encoder.weight"),
        ("action_encoder.bias", "action_encoder.bias"),
    ]

    @pytest.mark.parametrize("v5_name,v4_name", FAITHFUL_PAIRS)
    def test_faithful_layer_exact_equality(self, loaded_v5_and_v4, v5_name, v4_name):
        import mlx.nn as nn

        v5, v4, _ = loaded_v5_and_v4
        v5_params = dict(nn.utils.tree_flatten(v5.trainable_parameters()))
        v4_params = dict(nn.utils.tree_flatten(v4.trainable_parameters()))
        assert v5_name in v5_params, f"V5 missing param {v5_name}"
        assert v4_name in v4_params, f"V4 missing param {v4_name}"
        a = np.asarray(v5_params[v5_name])
        b = np.asarray(v4_params[v4_name])
        # Shapes must match exactly.
        assert a.shape == b.shape, (
            f"{v5_name} shape {a.shape} != V4 {v4_name} shape {b.shape}"
        )
        # EXACT byte equality (the binding gate -- tight tolerance).
        np.testing.assert_array_equal(a, b, err_msg=(
            f"FAITHFUL layer {v5_name} != V4 {v4_name} (must be EXACT after load)"
        ))
        # And allclose with a tight tolerance (redundant belt-and-braces).
        assert np.allclose(a, b, atol=1e-7, rtol=0.0), (
            f"FAITHFUL layer {v5_name} not allclose to V4 {v4_name} within 1e-7"
        )

    def test_shape_compat_layers_loaded_from_v4(self, loaded_v5_and_v4):
        """The SHAPE-COMPAT-DISCONNECTED layers (state_fuser.layers.2,
        candidate_scorer, value_head) ARE copied from V4 (by shape) -- their
        weights match the V4 source even though the inputs they receive differ.
        This documents that the copy happened (the non-parity comes from the
        fused INPUT, not from a failed copy)."""
        import mlx.nn as nn

        v5, v4, _ = loaded_v5_and_v4
        v5_params = dict(nn.utils.tree_flatten(v5.trainable_parameters()))
        v4_params = dict(nn.utils.tree_flatten(v4.trainable_parameters()))
        compat_pairs = [
            ("state_fuser.layers.2.weight", "state_encoder.layers.2.weight"),
            ("state_fuser.layers.2.bias", "state_encoder.layers.2.bias"),
            ("candidate_scorer.weight", "candidate_scorer.weight"),
            ("candidate_scorer.bias", "candidate_scorer.bias"),
            ("value_head.weight", "value_head.weight"),
            ("value_head.bias", "value_head.bias"),
        ]
        for v5n, v4n in compat_pairs:
            a = np.asarray(v5_params[v5n])
            b = np.asarray(v4_params[v4n])
            np.testing.assert_array_equal(a, b, err_msg=(
                f"shape-compat layer {v5n} not copied from V4 {v4n}"
            ))


# ============================================================================
# (c) Non-parity: full V5 forward does NOT match V4-Max (documents PARTIAL)
# ============================================================================
class TestNonParityFullLogitsDiffer:
    """Q3 PARTIAL: the warm-started V5 forward does NOT reproduce V4-Max 601
    logits. The fresh ``state_fuser.layers.0 (544,256)`` + extra SiLU break the
    V4 trunk, so even with faithful base_encoder + action_encoder, the fused
    state_emb differs and the 601 logits diverge. This PREVENTS the false
    confidence that warm-start = parity."""

    def test_full_logits_differ_beyond_tolerance(self, loaded_v5_and_v4):
        import mlx.core as mx

        from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V1_DIM, OBS_V5_DIM

        v5, v4, _ = loaded_v5_and_v4

        # Frozen obs (fixed seed). The V5 base-1456 prefix is IDENTICAL to the V4
        # obs, so the faithful base_encoder sees the same input in both. The V5
        # extra dims (global 32 + private 2400 + history 3240) are random, fed to
        # the FRESH encoders -- the divergence source.
        np.random.seed(123)
        v5_obs_np = (np.random.randn(1, OBS_V5_DIM).astype(np.float32) * 0.1)
        v4_obs_np = v5_obs_np[:, :OBS_V1_DIM].copy()
        # Same action_features for both (action_encoder is faithful -> same input).
        af_np = (np.random.randn(1, 601, ACTION_FEATURE_DIM).astype(np.float32) * 0.1)

        v4_obs = mx.array(v4_obs_np)
        v5_obs = mx.array(v5_obs_np)
        af = mx.array(af_np)

        v4_logits, v4_value = v4(v4_obs, af)
        v5_logits, v5_value, _ = v5(v5_obs, af)
        mx.eval(v4_logits, v5_logits, v4_value, v5_value)

        v4l = np.asarray(v4_logits)
        v5l = np.asarray(v5_logits)
        assert v4l.shape == (1, 601) and v5l.shape == (1, 601), (
            f"logit shape mismatch v4={v4l.shape} v5={v5l.shape}"
        )

        # The 601 logits must NOT match V4-Max within tolerance (documents PARTIAL).
        assert not np.allclose(v5l, v4l, atol=1e-3, rtol=1e-3), (
            "V5 warm-started logits MUST NOT match V4-Max logits (Q3 partial: the "
            "fresh state_fuser.layers.0 + extra SiLU break the trunk). "
            f"v5[0,:5]={v5l[0,:5]} v4[0,:5]={v4l[0,:5]}"
        )
        # And the divergence is meaningful (not numerical noise).
        max_diff = float(np.max(np.abs(v5l - v4l)))
        assert max_diff > 0.05, (
            f"logit divergence too small ({max_diff:.6f}) -- expected a "
            "meaningful difference from the architectural disconnect, not noise"
        )

    def test_base_encoder_path_is_faithful_despite_logit_divergence(
        self, loaded_v5_and_v4
    ):
        """The divergence is NOT a failed faithful copy: feeding the SAME
        base-1456 input through V5's base_encoder.layers.0 (faithful) and V4's
        state_encoder.layers.0 produces IDENTICAL first-layer outputs. The logit
        divergence comes from the downstream fused path, not the faithful
        layers. This isolates WHAT is faithful vs what diverges."""
        import mlx.core as mx

        from train_v3.contracts import OBS_V1_DIM

        v5, v4, _ = loaded_v5_and_v4
        np.random.seed(7)
        base = mx.array(np.random.randn(2, OBS_V1_DIM).astype(np.float32) * 0.1)

        # V5 base_encoder = Sequential(Linear(1456,256), SiLU) -- the first
        # Linear (layers[0]) is faithful to V4 state_encoder.layers[0].
        v5_base_lin = v5.base_encoder.layers[0](base)
        # V4 state_encoder.layers[0] (the first Linear in state_encoder).
        v4_state_lin = v4.state_encoder.layers[0](base)
        mx.eval(v5_base_lin, v4_state_lin)
        np.testing.assert_array_equal(
            np.asarray(v5_base_lin), np.asarray(v4_state_lin), err_msg=(
                "base_encoder.layers.0 must produce IDENTICAL output to V4 "
                "state_encoder.layers.0 for the same input (faithful path)"
            )
        )


# ============================================================================
# (d) strict=False: no silent key drops
# ============================================================================
class TestStrictFalseNoSilentDrops:
    """``load_v4_max_into_v5`` is strict=False semantics: every mapped V4 weight
    key is transferred; no weight key is silently dropped. The V4 ``_opt_`` Adam
    optimizer-state keys are NOT loaded (by design -- warm-start is a policy
    transfer, not an optimizer restore) and are documented in the report."""

    def test_no_weight_key_drops(self, loaded_v5_and_v4):
        from train_v3.warm_start_v5 import TRANSFER_MAP

        _, _, report = loaded_v5_and_v4
        # Zero weight-key drops on success.
        assert report["dropped"] == [], (
            f"unexpected dropped weight keys: {report['dropped']}"
        )
        # Every V4 weight key in the npz is a V4 source in the transfer map (no
        # V4 weight key is left unmapped -> no silent drop).
        v4_sources = {src for (src, _cat) in TRANSFER_MAP.values()}
        for v4_key in report["v4_weight_keys"]:
            assert v4_key in v4_sources, (
                f"V4 weight key {v4_key} has no V5 target in TRANSFER_MAP "
                "(silent drop risk)"
            )

    def test_all_expected_v4_weights_transferred(self, loaded_v5_and_v4):
        from train_v3.warm_start_v5 import V4_WEIGHT_KEYS_EXPECTED

        _, _, report = loaded_v5_and_v4
        transferred_layers = {t["v4_source"] for t in report["transferred"]}
        # All 10 expected V4 weight keys must appear in the transferred set.
        for k in V4_WEIGHT_KEYS_EXPECTED:
            assert k in transferred_layers, (
                f"expected V4 weight {k} not in transferred set: "
                f"{sorted(transferred_layers)}"
            )
        assert len(report["transferred"]) == len(V4_WEIGHT_KEYS_EXPECTED), (
            f"transferred count {len(report['transferred'])} != expected "
            f"{len(V4_WEIGHT_KEYS_EXPECTED)}"
        )

    def test_opt_keys_documented_not_loaded_by_design(self, loaded_v5_and_v4):
        """The V4 _opt_ Adam optimizer-state keys are listed in the report as
        skipped (warm-start transfers POLICY weights, not optimizer state)."""
        _, _, report = loaded_v5_and_v4
        opt_keys = report["v4_opt_keys_skipped"]
        assert len(opt_keys) > 0, "V4-Max npz should contain _opt_ Adam-state keys"
        for k in opt_keys:
            assert k.startswith("_opt_"), f"non-opt key in opt-skip list: {k}"
        # The _opt_ keys are NOT in the transferred set (not loaded into V5).
        transferred_layers = {t["v5_layer"] for t in report["transferred"]}
        for k in opt_keys:
            assert k not in transferred_layers

    def test_fresh_layers_left_untouched(self, loaded_v5_and_v4):
        """The 5 FRESH V5 layers (no V4 counterpart) are listed and were NOT
        overwritten by V4 (they keep their seeded default init)."""
        import mlx.nn as nn

        from train_v3.warm_start_v5 import FRESH_V5_LAYERS, TRANSFER_MAP

        v5, _, report = loaded_v5_and_v4
        # The report lists the fresh layers.
        assert set(report["fresh"]) == set(FRESH_V5_LAYERS)
        # No fresh layer param is a key in TRANSFER_MAP (it was never a load
        # target).
        for fresh_layer in FRESH_V5_LAYERS:
            for suffix in (".weight", ".bias"):
                assert (fresh_layer + suffix) not in TRANSFER_MAP, (
                    f"fresh layer {fresh_layer}{suffix} must not be in TRANSFER_MAP"
                )
        # The fresh params exist and are finite (default init, not corrupted).
        v5_params = dict(nn.utils.tree_flatten(v5.trainable_parameters()))
        for fresh_layer in FRESH_V5_LAYERS:
            for suffix in (".weight", ".bias"):
                key = fresh_layer + suffix
                assert key in v5_params, f"fresh param {key} missing from V5"
                arr = np.asarray(v5_params[key])
                assert np.all(np.isfinite(arr)), f"fresh param {key} not finite"


# ============================================================================
# (e) npz keys dumped match the V4 architecture (model_mlx.py:37-81)
# ============================================================================
class TestNpzKeysDumped:
    """The V4-Max npz key/shape list must match the ``model_mlx.ActionConditionedPolicy``
    architecture (ai/train_v2/model_mlx.py:37-81): state_encoder (Sequential of
    2 Linear: layers.0 (1456->256) + layers.2 (256->256)), action_encoder
    (171->128), candidate_scorer (384->1), value_head (256->1), plus _opt_ Adam
    state + __meta__."""

    def test_npz_weight_keys_match_v4_architecture(self, v4_max_npz_path):
        data = np.load(v4_max_npz_path, allow_pickle=False)
        try:
            keys = list(data.files)
            # Every expected V4 weight key is present with the exact shape.
            for k, expected_shape in EXPECTED_V4_WEIGHT_SHAPES.items():
                assert k in keys, f"V4 weight key {k} missing from npz"
                assert tuple(data[k].shape) == expected_shape, (
                    f"V4 key {k} shape {tuple(data[k].shape)} != expected "
                    f"{expected_shape}"
                )
            # The weight keys are exactly the expected set (no extra weight keys
            # beyond _opt_ + __meta__).
            weight_keys = {
                k for k in keys if not k.startswith("_opt_") and k != "__meta__"
            }
            assert weight_keys == set(EXPECTED_V4_WEIGHT_SHAPES.keys()), (
                f"unexpected V4 weight keys: {weight_keys ^ set(EXPECTED_V4_WEIGHT_SHAPES)}"
            )
        finally:
            data.close()

    def test_npz_has_opt_state_and_meta(self, v4_max_npz_path):
        """The V4-Max npz also carries _opt_ Adam optimizer state (m/v moments +
        step + learning_rate) and a __meta__ JSON buffer -- these document the
        checkpoint provenance but are NOT loaded by the warm-start."""
        data = np.load(v4_max_npz_path, allow_pickle=False)
        try:
            keys = list(data.files)
            opt_keys = [k for k in keys if k.startswith("_opt_")]
            assert len(opt_keys) >= 10, (
                f"expected Adam-state _opt_ keys, got {len(opt_keys)}: {opt_keys}"
            )
            # __meta__ is a uint8 JSON buffer (model_mlx.save_checkpoint:112).
            assert "__meta__" in keys, "npz missing __meta__"
            meta = data["__meta__"]
            assert meta.dtype == np.uint8, f"__meta__ dtype {meta.dtype} != uint8"
            import json

            meta_dict = json.loads(meta.tobytes().decode("utf-8"))
            assert meta_dict.get("model_version") == "classic_action_conditioned_mlx_v1"
            assert meta_dict.get("obs_dim") == 1456
            assert meta_dict.get("action_feature_dim") == 171
            assert meta_dict.get("max_candidate_actions") == 601
        finally:
            data.close()

    def test_npz_dtypes_float32_weights(self, v4_max_npz_path):
        """All V4 weight/bias arrays are float32 (the MLX training dtype)."""
        data = np.load(v4_max_npz_path, allow_pickle=False)
        try:
            for k, _shape in EXPECTED_V4_WEIGHT_SHAPES.items():
                assert data[k].dtype == np.float32, (
                    f"V4 weight {k} dtype {data[k].dtype} != float32"
                )
        finally:
            data.close()