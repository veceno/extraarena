"""Block A component A2 — tests for ``train_v3.bc_train`` (BC training loop).

TRACKED (``git check-ignore`` exit=1 — NOT gitignored). Synthetic
``BCTransition`` tuples are constructed DIRECTLY (no real pilot trace needed) so
the tests are unit-level + deterministic. MLX is importable in this worktree
(``pytest.importorskip("mlx")`` is a no-op here; guards other envs).

The warm-started tests run against a REAL warm-started policy when the V4-Max
npz is reachable (it is — the main-repo absolute path); they skip-on-npz-absent
like ``test_train_v2_warm_start_v5.py`` (the npz is gitignored in worktrees).

Tests (spec A2 acceptance):
  1. ``test_loss_decreases`` — a few BC steps on a synthetic BCTransition batch
     -> mean total loss strictly decreases.
  2. ``test_faithful_layers_preserved_after_bc`` — with ``freeze_faithful=True``,
     after K BC steps the FAITHFUL params (``base_encoder.layers.0`` +
     ``action_encoder``) are BYTE-IDENTICAL to their post-warm-start values
     (BC did NOT touch them), while the trainable params (``candidate_scorer``)
     DID move.
  3. ``test_mana_draw_head_learns_signal`` — a synthetic batch where
     mana_draw-legal rows have ``is_mana_draw=True`` -> after a few BC steps the
     mana_draw head sigmoid(logit) INCREASES for those rows (BCE pulls toward 1).
  4. ``test_skip_if_no_npz`` — when the V4-Max npz is unavailable
     (``resolve_v4_max_npz_path`` raises), BC handles it gracefully: warm-start
     is SKIPPED and BC runs on a fresh-init policy WITHOUT crashing.
  5. ``test_checkpoint_round_trip`` — the BC-seed checkpoint (via
     ``model_mlx.save_checkpoint``) loads back into a fresh
     ``V5ActionConditionedPolicy`` with BYTE-IDENTICAL params + identical forward
     (PPO A4 resumes via the SAME loader).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# MLX is importable in this worktree; importorskip guards other envs (no-op here).
pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from ai.train_v2.model_mlx import load_checkpoint, save_checkpoint  # noqa: E402
from train_v3.bc_dataset import BCTransition  # noqa: E402
from train_v3.bc_train import (  # noqa: E402
    FAITHFUL_PARAM_NAMES,
    assert_frozen_preserved,
    compute_bc_loss,
    frozen_param_names,
    prepare_bc_policy,
    run_bc_training,
    snapshot_frozen_params,
    train_bc,
)
from train_v3.contracts import ACTION_FEATURE_DIM, MAX_CANDIDATE_ACTIONS, OBS_V5_DIM  # noqa: E402
from train_v3.v5_policy import V5ActionConditionedPolicy  # noqa: E402

# Canonical V4-Max npz (gitignored in worktrees; reachable via the main-repo
# absolute path). Mirrors ``test_train_v2_warm_start_v5.py:31``.
DEFAULT_V4_MAX_NPZ = (
    "/Users/laveqox/Documents/ExtraArenaRaS/ai/train_v2/runs/"
    "m4_balanced_from_0950_20260522_144431/checkpoints/update_1190.npz"
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def v4_max_npz_path():
    """Resolve the V4-Max npz; skip the whole module if absent (gitignored)."""
    from train_v3.warm_start_v5 import resolve_v4_max_npz_path

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
def warmed_policy(v4_max_npz_path):
    """A deterministic V5 policy warm-started from the V4-Max npz."""
    from train_v3.warm_start_v5 import load_v4_max_into_v5

    mx.random.seed(42)
    policy = V5ActionConditionedPolicy()
    load_v4_max_into_v5(policy, npz_path=v4_max_npz_path)
    return policy


@pytest.fixture()
def fresh_policy():
    """A deterministic fresh-init V5 policy (no warm-start)."""
    mx.random.seed(42)
    return V5ActionConditionedPolicy()


# ---------------------------------------------------------------------------
# Synthetic BCTransition builders (no real pilot trace needed).
# ---------------------------------------------------------------------------
def make_transition(
    *,
    target_tcode,
    is_mana_draw: bool,
    mana_draw_legal: bool,
    legal_ids,
    rng: np.random.Generator,
    terminal: bool = False,
) -> BCTransition:
    """Build one synthetic ``BCTransition`` directly.

    ``legal_ids`` is the set of candidate ids that are legal for this row; the
    target (when not None) MUST be in ``legal_ids`` (A1 guarantee).
    """
    obs = rng.standard_normal(OBS_V5_DIM).astype(np.float32)
    af = rng.standard_normal((MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM)).astype(np.float32)
    legal_mask = np.zeros(MAX_CANDIDATE_ACTIONS, dtype=bool)
    for cid in legal_ids:
        legal_mask[int(cid)] = True
    return BCTransition(
        obs=obs,
        action_features=af,
        target_tcode=target_tcode,
        is_mana_draw=is_mana_draw,
        mana_draw_legal=mana_draw_legal,
        legal_mask=legal_mask,
        reward=0.0,
        terminal=terminal,
        meta={"decision_source": "human", "action_type": "mana_draw" if is_mana_draw else "play_card"},
    )


def make_candidate_batch(rng: np.random.Generator, n: int = 8, target: int = 0):
    """A batch of normal-action rows all targeting the same candidate id.

    The target is legal (in legal_ids); a couple of distractor candidates are
    also legal so the CE has a real distribution to sharpen.
    """
    rows = []
    for _ in range(n):
        legal_ids = {target, (target + 7) % MAX_CANDIDATE_ACTIONS, (target + 13) % MAX_CANDIDATE_ACTIONS}
        rows.append(
            make_transition(
                target_tcode=target,
                is_mana_draw=False,
                mana_draw_legal=False,
                legal_ids=legal_ids,
                rng=rng,
            )
        )
    return rows


def make_mana_draw_batch(rng: np.random.Generator, n: int = 8):
    """A batch of mana_draw rows: legal + is_mana_draw=True (BCE target 1.0)."""
    rows = []
    for _ in range(n):
        # mana_draw is OUTSIDE the 601 space; legal_mask can be empty (only
        # mana_draw available) — target_tcode=None.
        rows.append(
            make_transition(
                target_tcode=None,
                is_mana_draw=True,
                mana_draw_legal=True,
                legal_ids=set(),
                rng=rng,
            )
        )
    return rows


def _set_mana_draw_head(policy, *, weight_val: float = 0.0, bias_val: float = -6.0):
    """Set the mana_draw head to a KNOWN initial state so BC tests are
    deterministic regardless of seed.

    With ``weight_val=0`` the head logit is EXACTLY ``bias_val`` for every row
    (independent of state_emb), giving a known sigmoid starting point. This
    lets the learn-signal test guarantee the head starts AWAY from the target
    so BCE has room to pull it, and lets the BCE-mask test compute the expected
    loss analytically.
    """
    flat = nn.utils.tree_flatten(policy.parameters())
    new_flat = []
    for name, val in flat:
        if name == "mana_draw_head.weight":
            new_flat.append((name, mx.full(val.shape, weight_val, dtype=val.dtype)))
        elif name == "mana_draw_head.bias":
            new_flat.append((name, mx.full(val.shape, bias_val, dtype=val.dtype)))
        else:
            new_flat.append((name, val))
    policy.update(nn.utils.tree_unflatten(new_flat))
    mx.eval(policy.parameters())


# ---------------------------------------------------------------------------
# (1) test_loss_decreases
# ---------------------------------------------------------------------------
def test_loss_decreases(warmed_policy):
    """A few BC steps on a synthetic candidate batch -> total loss decreases."""
    rng = np.random.default_rng(0)
    dataset = make_candidate_batch(rng, n=8, target=0)

    report = train_bc(
        warmed_policy,
        dataset,
        freeze_faithful=True,
        learning_rate=1e-2,
        steps=20,
        rng=np.random.default_rng(1),
    )
    losses = report["losses"]
    assert len(losses) == 20
    # Strict, meaningful decrease (BC converges fast on a single-target batch).
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"
    assert losses[-1] < losses[0] * 0.5, (
        f"loss decrease too small: {losses[0]} -> {losses[-1]}"
    )
    # All finite.
    assert all(np.isfinite(x) for x in losses), f"non-finite loss: {losses}"


def test_loss_decreases_fresh(fresh_policy):
    """Same as above on a FRESH-init policy (no warm-start) — BC still works."""
    rng = np.random.default_rng(2)
    dataset = make_candidate_batch(rng, n=8, target=5)

    report = train_bc(
        fresh_policy,
        dataset,
        freeze_faithful=True,
        learning_rate=1e-2,
        steps=20,
        rng=np.random.default_rng(3),
    )
    losses = report["losses"]
    assert losses[-1] < losses[0] * 0.5, (
        f"fresh-init BC loss decrease too small: {losses[0]} -> {losses[-1]}"
    )


# ---------------------------------------------------------------------------
# (2) test_faithful_layers_preserved_after_bc
# ---------------------------------------------------------------------------
def test_faithful_layers_preserved_after_bc(warmed_policy):
    """freeze_faithful=True -> FAITHFUL params byte-identical after BC; the
    trainable candidate_scorer DID move."""
    # Snapshot the FAITHFUL params right after warm-start (pre-BC).
    faithful_before = snapshot_frozen_params(warmed_policy, FAITHFUL_PARAM_NAMES)
    # Snapshot candidate_scorer (a trainable param) to prove BC actually ran.
    cs_before = snapshot_frozen_params(
        warmed_policy,
        frozenset({"candidate_scorer.weight", "candidate_scorer.bias"}),
    )

    rng = np.random.default_rng(4)
    dataset = make_candidate_batch(rng, n=8, target=0)
    report = train_bc(
        warmed_policy,
        dataset,
        freeze_faithful=True,
        learning_rate=1e-2,
        steps=15,
        rng=np.random.default_rng(5),
    )

    # FAITHFUL params byte-identical (the zero-grad idiom guarantees this).
    assert_frozen_preserved(warmed_policy, faithful_before)

    # Also assert via the report's own frozen snapshot (pre-BC) for independence.
    for name in FAITHFUL_PARAM_NAMES:
        before = report["frozen_snapshot"][name]
        after = np.array(dict(nn.utils.tree_flatten(warmed_policy.trainable_parameters()))[name])
        assert np.array_equal(before, after), f"faithful param {name} moved during BC"

    # Trainable candidate_scorer DID move (proves BC ran + the freeze is
    # targeted, not a blanket no-op).
    cs_after = snapshot_frozen_params(
        warmed_policy,
        frozenset({"candidate_scorer.weight", "candidate_scorer.bias"}),
    )
    moved = any(not np.array_equal(cs_before[k], cs_after[k]) for k in cs_before)
    assert moved, "candidate_scorer did not move — BC did not actually train"

    # The frozen set reported matches the spec (FAITHFUL + state_fuser.layers.2 + value_head).
    frozen = set(report["frozen_param_names"])
    assert FAITHFUL_PARAM_NAMES <= frozen, "FAITHFUL params not in frozen set"
    assert "state_fuser.layers.2.weight" in frozen
    assert "value_head.weight" in frozen


def test_freeze_faithful_false_unfreezes_faithful(fresh_policy):
    """freeze_faithful=False (ablation) -> FAITHFUL params ARE trainable and DO
    move during BC (the override unfreezes them)."""
    faithful_before = snapshot_frozen_params(fresh_policy, FAITHFUL_PARAM_NAMES)
    rng = np.random.default_rng(6)
    dataset = make_candidate_batch(rng, n=8, target=0)
    report = train_bc(
        fresh_policy,
        dataset,
        freeze_faithful=False,
        learning_rate=1e-2,
        steps=10,
        rng=np.random.default_rng(7),
    )
    assert FAITHFUL_PARAM_NAMES.isdisjoint(set(report["frozen_param_names"])), (
        "freeze_faithful=False should unfreeze FAITHFUL params"
    )
    faithful_after = snapshot_frozen_params(fresh_policy, FAITHFUL_PARAM_NAMES)
    moved = any(
        not np.array_equal(faithful_before[k], faithful_after[k])
        for k in faithful_before
    )
    assert moved, "freeze_faithful=False should let FAITHFUL params move"


# ---------------------------------------------------------------------------
# (3) test_mana_draw_head_learns_signal
# ---------------------------------------------------------------------------
def test_mana_draw_head_learns_signal(warmed_policy):
    """mana_draw-legal rows with is_mana_draw=True -> BCE pulls the head sigmoid
    UP for those rows after a few BC steps.

    The head is RESET to a known low-sigmoid state (logit=-6 -> sigmoid~0.0025)
    so the test is deterministic regardless of seed and the BCE has room to pull
    toward the 1.0 target. With freeze_faithful=True the mana_draw_head is in the
    trainable move set, so it DOES move."""
    _set_mana_draw_head(warmed_policy, weight_val=0.0, bias_val=-6.0)

    rng = np.random.default_rng(8)
    dataset = make_mana_draw_batch(rng, n=8)

    from train_v3.bc_train import collate_bc_batch

    probe = collate_bc_batch(dataset)
    _, _, md_logit_before = warmed_policy(
        probe["obs"], probe["action_features"], mana_draw_legal=None
    )
    sig_before = float(mx.mean(mx.sigmoid(md_logit_before)))
    # Sanity: the reset puts the head well below the 1.0 target.
    assert sig_before < 0.01, f"setup: head should start low, got {sig_before}"

    report = train_bc(
        warmed_policy,
        dataset,
        freeze_faithful=True,
        learning_rate=1e-2,
        steps=20,
        rng=np.random.default_rng(9),
    )
    # BCE should decrease (the head is learning the 1.0 target from a low start).
    bce = report["mana_draw_bce"]
    assert bce[-1] < bce[0], f"mana_draw BCE did not decrease: {bce[0]} -> {bce[-1]}"

    _, _, md_logit_after = warmed_policy(
        probe["obs"], probe["action_features"], mana_draw_legal=None
    )
    sig_after = float(mx.mean(mx.sigmoid(md_logit_after)))
    assert sig_after > sig_before, (
        f"mana_draw sigmoid did not increase for legal-True rows: "
        f"{sig_before} -> {sig_after}"
    )
    # Meaningful learning (not just a tiny nudge).
    assert sig_after > 0.5, (
        f"mana_draw head learned too little: sigmoid {sig_before} -> {sig_after}"
    )


def test_mana_draw_bce_only_on_legal_rows(warmed_policy):
    """The mana_draw BCE LOSS VALUE depends only on LEGAL rows (mirrors
    ``mana_draw_head_v5.select_includes_mana_draw:116`` — the head only matters
    when mana_draw is legal).

    This verifies the LOSS MASK (illegal rows excluded from the BCE sum/mean),
    NOT the illegal rows' output (which DOES move because the head weights are
    SHARED and trained on legal-row gradients — that is expected). With the head
    reset to a known logit, the expected BCE over legal rows is analytic."""
    from train_v3.bc_train import collate_bc_batch, compute_bc_loss_terms

    _set_mana_draw_head(warmed_policy, weight_val=0.0, bias_val=-2.0)
    # logit = -2 for every row -> sigmoid = 0.1192 -> BCE(target=1) = -log(0.1192)

    rng = np.random.default_rng(10)
    # 4 rows: [legal+md=True, legal+md=True, ILLEGAL+md=False, ILLEGAL+md=False]
    rows = []
    for _ in range(2):
        rows.append(
            make_transition(
                target_tcode=None, is_mana_draw=True, mana_draw_legal=True,
                legal_ids=set(), rng=rng,
            )
        )
    for _ in range(2):
        rows.append(
            make_transition(
                target_tcode=None, is_mana_draw=False, mana_draw_legal=False,
                legal_ids=set(), rng=rng,
            )
        )
    batch = collate_bc_batch(rows)
    _, metrics = compute_bc_loss_terms(warmed_policy, batch)
    md_bce = float(metrics["mana_draw_bce"])

    # Expected: mean over the 2 LEGAL rows of BCE(sigmoid(-2), target=1.0).
    # Illegal rows are EXCLUDED from the sum/mean (md_legal_count=2).
    p = float(mx.sigmoid(mx.array(-2.0)))
    p = min(max(p, 1e-7), 1.0 - 1e-7)
    expected = -float(np.log(p))  # target=1.0 for both legal rows
    assert np.isclose(md_bce, expected, atol=1e-4), (
        f"mana_draw BCE should equal mean over LEGAL rows only: "
        f"got {md_bce}, expected {expected}"
    )
    assert int(metrics["mana_draw_legal_rows"]) == 2, "should count 2 legal rows"

    # A batch with ALL rows mana_draw-ILLEGAL -> BCE is 0 (no legal rows
    # contribute; guarded against divide-by-zero).
    rows_all_illegal = [
        make_transition(
            target_tcode=None, is_mana_draw=False, mana_draw_legal=False,
            legal_ids=set(), rng=rng,
        )
        for _ in range(3)
    ]
    batch2 = collate_bc_batch(rows_all_illegal)
    _, metrics2 = compute_bc_loss_terms(warmed_policy, batch2)
    assert float(metrics2["mana_draw_bce"]) == 0.0, (
        "all-illegal batch should have zero mana_draw BCE"
    )


# ---------------------------------------------------------------------------
# (4) test_skip_if_no_npz
# ---------------------------------------------------------------------------
def test_skip_if_no_npz(monkeypatch):
    """When the V4-Max npz is unavailable (resolve_v4_max_npz_path raises),
    BC handles it gracefully: warm-start is SKIPPED and BC runs on a
    fresh-init policy WITHOUT crashing.

    Chosen behavior (documented in bc_train.prepare_bc_policy): skip the
    warm-start, run BC on a fresh-init policy, record ``warm_started=False`` +
    the skip reason — do NOT crash. MLX itself is present (no mlx gate)."""
    import train_v3.warm_start_v5 as wsm

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated: V4-Max npz not found (test)")

    monkeypatch.setattr(wsm, "resolve_v4_max_npz_path", _raise)

    policy, warm_report = prepare_bc_policy(npz_path=None)
    assert warm_report["warm_started"] is False
    assert warm_report["skip_reason"]
    assert warm_report["npz_path"] is None
    # The policy is a fresh V5ActionConditionedPolicy (usable for BC).
    assert isinstance(policy, V5ActionConditionedPolicy)

    # BC still runs on the fresh-init policy — no crash, finite loss.
    rng = np.random.default_rng(12)
    dataset = make_candidate_batch(rng, n=6, target=0)
    report = train_bc(
        policy, dataset, freeze_faithful=True, learning_rate=1e-2, steps=10,
        rng=np.random.default_rng(13),
    )
    assert all(np.isfinite(x) for x in report["losses"]), report["losses"]
    assert report["losses"][-1] < report["losses"][0], (
        "fresh-init BC (no warm-start) should still decrease loss"
    )

    # The full pipeline (run_bc_training) also handles the skip gracefully.
    policy2, full_report = run_bc_training(
        dataset, npz_path=None, freeze_faithful=True, learning_rate=1e-2, steps=5,
        rng=np.random.default_rng(14),
    )
    assert full_report["warm_start"]["warm_started"] is False
    assert len(full_report["training"]["losses"]) == 5


def test_skip_if_no_npz_via_env(monkeypatch):
    """Skip-gate also triggers when V4_MAX_NPZ_PATH points to a non-existent
    file (the env branch of resolve_v4_max_npz_path raises)."""
    monkeypatch.setenv("V4_MAX_NPZ_PATH", "/nonexistent/path/to/missing.npz")
    policy, warm_report = prepare_bc_policy(npz_path=None)
    assert warm_report["warm_started"] is False
    assert isinstance(policy, V5ActionConditionedPolicy)


# ---------------------------------------------------------------------------
# (5) test_checkpoint_round_trip
# ---------------------------------------------------------------------------
def test_checkpoint_round_trip(warmed_policy, tmp_path):
    """The BC-seed checkpoint (save_checkpoint) loads back into a fresh
    V5ActionConditionedPolicy with byte-identical params + identical forward
    (PPO A4 resumes via the SAME load_checkpoint loader)."""
    rng = np.random.default_rng(15)
    dataset = make_candidate_batch(rng, n=6, target=0)
    train_bc(
        warmed_policy, dataset, freeze_faithful=True, learning_rate=1e-2, steps=5,
        rng=np.random.default_rng(16),
    )

    # Snapshot params + a forward output BEFORE saving.
    params_before = {
        n: np.array(v) for n, v in nn.utils.tree_flatten(warmed_policy.trainable_parameters())
    }
    probe_obs = mx.array(np.random.default_rng(17).standard_normal((4, OBS_V5_DIM)).astype(np.float32))
    probe_af = mx.array(
        np.random.default_rng(18).standard_normal((4, MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM)).astype(np.float32)
    )
    fwd_before = warmed_policy(probe_obs, probe_af, mana_draw_legal=None)
    fwd_before_np = tuple(np.array(t) for t in fwd_before)

    ckpt = str(tmp_path / "bc_seed.npz")
    save_checkpoint(
        ckpt, warmed_policy, optimizer=None,
        metadata={"kind": "bc_seed", "policy_kind": "v5_split_encoder", "step": 5},
    )

    # Load into a FRESH policy (the PPO-A4 resume path).
    reloaded = V5ActionConditionedPolicy()
    meta = load_checkpoint(ckpt, reloaded)
    assert meta["metadata"]["kind"] == "bc_seed"

    params_after = {
        n: np.array(v) for n, v in nn.utils.tree_flatten(reloaded.trainable_parameters())
    }
    assert set(params_before) == set(params_after), "param key set changed on round-trip"
    for k in params_before:
        assert np.array_equal(params_before[k], params_after[k]), (
            f"param {k} not byte-identical after checkpoint round-trip"
        )

    # Identical forward.
    fwd_after = reloaded(probe_obs, probe_af, mana_draw_legal=None)
    fwd_after_np = tuple(np.array(t) for t in fwd_after)
    for a, b in zip(fwd_before_np, fwd_after_np):
        assert np.allclose(a, b, atol=1e-6), "forward output changed on round-trip"


def test_run_bc_training_writes_checkpoint(warmed_policy, v4_max_npz_path, tmp_path):
    """run_bc_training writes a BC-seed checkpoint consumable by PPO A4."""
    rng = np.random.default_rng(19)
    dataset = make_candidate_batch(rng, n=6, target=0)
    ckpt = str(tmp_path / "bc_seed_full.npz")
    _, report = run_bc_training(
        dataset, npz_path=v4_max_npz_path, freeze_faithful=True,
        learning_rate=1e-2, steps=5, checkpoint_path=ckpt,
        rng=np.random.default_rng(20),
    )
    assert Path(ckpt).is_file(), "checkpoint not written"
    assert report["warm_start"]["warm_started"] is True
    assert report["checkpoint_meta"]["kind"] == "bc_seed"
    assert report["checkpoint_meta"]["policy_kind"] == "v5_split_encoder"
    assert report["checkpoint_meta"]["warm_started"] is True
    # Reloads into a fresh policy (PPO A4 resume path).
    reloaded = V5ActionConditionedPolicy()
    load_checkpoint(ckpt, reloaded)
    # Faithful layers in the reloaded policy still byte-match the V4-Max
    # source (BC preserved them).
    with np.load(v4_max_npz_path, allow_pickle=False) as data:
        v4_base_w = data["state_encoder.layers.0.weight"]
    flat = dict(nn.utils.tree_flatten(reloaded.trainable_parameters()))
    assert np.array_equal(np.array(flat["base_encoder.layers.0.weight"]), v4_base_w), (
        "faithful base_encoder.layers.0.weight lost on checkpoint round-trip"
    )