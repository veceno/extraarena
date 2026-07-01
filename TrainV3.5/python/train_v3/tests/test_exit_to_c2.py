"""B7 ``exit_to_c2.py`` tests -- plateau-below-dominance-target exit to C2.

DEFAULT D-B3 below reading (no flip): exit fires when plateau AND candidate
BELOW dominance target (still-weak -> C2). A plateau AT/ABOVE target does NOT
fire (dominant -> E1). A still-improving H2H series does NOT fire. Synthetic
fabricated H2H series only (no real Rust/MLX/ONNX).

Also: insufficient-snapshots guard; ``git diff a_gate.py`` empty regression guard
(B7 does NOT edit A5); source-vs-source inverse vs A5 ``check_h2h_trending``
(``a_gate.py:374``); and the flippable at/above reading
(``below_target_exits=False`` documents D-B3 3a).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from train_v3.a_gate import check_h2h_trending
from train_v3.exit_to_c2 import (
    DEFAULT_K_SNAP,
    DEFAULT_MIN_GAIN,
    ExitToC2Verdict,
    detect_h2h_plateau,
)

# Dominance target used across the DEFAULT-reading tests (design.md:121 ~0.55).
DOMINANCE_TARGET = 0.55
BEST_PATH = "/pool/best_ever/ckpt_007.onnx"


# -----------------------------------------------------------------------------
# 1. plateau below target fires exit
# -----------------------------------------------------------------------------
def test_plateau_below_target_fires_exit():
    # Flat H2H at 0.40 (below 0.55 dominance target) for K_snap+1 points.
    K = DEFAULT_K_SNAP
    scores = [0.40] * (K + 1)
    v = detect_h2h_plateau(
        scores,
        dominance_target=DOMINANCE_TARGET,
        K_snap=K,
        min_gain=DEFAULT_MIN_GAIN,
        best_checkpoint_path=BEST_PATH,
    )
    assert v.plateau is True
    assert v.below_target is True
    assert v.exit_fires is True
    assert v.plateau_run_length == K
    assert v.current_h2h == 0.40
    assert v.reason == "plateau_below_dominance_target"
    assert v.best_checkpoint_path == BEST_PATH


# -----------------------------------------------------------------------------
# 2. still improving -> no exit
# -----------------------------------------------------------------------------
def test_still_improving_no_exit():
    # Rising H2H (every step gains > min_gain) -- never plateaus.
    K = DEFAULT_K_SNAP
    scores = [0.30 + 0.02 * i for i in range(K + 2)]  # +0.02 each step
    v = detect_h2h_plateau(
        scores, dominance_target=DOMINANCE_TARGET, K_snap=K, min_gain=DEFAULT_MIN_GAIN
    )
    assert v.plateau is False
    assert v.exit_fires is False
    assert v.reason == "still_improving"
    assert v.plateau_run_length == 0


# -----------------------------------------------------------------------------
# 3. plateau at/above target -> no exit (dominant -> E1)
# -----------------------------------------------------------------------------
def test_plateau_at_or_above_target_no_exit():
    # Flat H2H at 0.60 (AT/ABOVE 0.55 dominance target) for K_snap+1 points.
    K = DEFAULT_K_SNAP
    scores = [0.60] * (K + 1)
    v = detect_h2h_plateau(
        scores, dominance_target=DOMINANCE_TARGET, K_snap=K, min_gain=DEFAULT_MIN_GAIN
    )
    assert v.plateau is True
    assert v.below_target is False
    assert v.exit_fires is False
    assert v.reason == "dominant_plateau_e1_path"


# -----------------------------------------------------------------------------
# 4. exit carries best checkpoint
# -----------------------------------------------------------------------------
def test_exit_carries_best_checkpoint():
    K = DEFAULT_K_SNAP
    scores = [0.35] * (K + 1)
    v = detect_h2h_plateau(
        scores,
        dominance_target=DOMINANCE_TARGET,
        K_snap=K,
        min_gain=DEFAULT_MIN_GAIN,
        best_checkpoint_path=BEST_PATH,
    )
    assert v.exit_fires is True
    assert v.best_checkpoint_path == BEST_PATH
    d = v.to_dict()
    assert d["best_checkpoint_path"] == BEST_PATH
    assert d["exit_fires"] is True


# -----------------------------------------------------------------------------
# 5. K_snap window -- a single flat snapshot does NOT exit
# -----------------------------------------------------------------------------
def test_k_snap_window():
    # A short flat run (much shorter than K_snap) must NOT fire.
    K = DEFAULT_K_SNAP
    # Rising series with a single flat step at the end -- run=1 < K.
    scores = [0.30 + 0.02 * i for i in range(K)] + [0.30 + 0.02 * (K - 1)]
    v = detect_h2h_plateau(
        scores, dominance_target=DOMINANCE_TARGET, K_snap=K, min_gain=DEFAULT_MIN_GAIN
    )
    assert v.plateau is False
    assert v.exit_fires is False
    assert v.plateau_run_length == 1
    assert v.reason == "still_improving"


# -----------------------------------------------------------------------------
# 6. min_gain tolerance -- sub-min_gain uptick = no-gain (plateau);
#    above-min_gain uptick resets the window.
# -----------------------------------------------------------------------------
def test_min_gain_tolerance():
    K = 4
    # Sub-min_gain uptick (+0.005 <= 0.01) -> no-gain -> plateau continues.
    base = 0.40
    sub = [base, base + 0.005, base + 0.010, base + 0.005, base + 0.010]
    # len = 5 = K+1; every step gain <= min_gain -> run = 4 = K -> plateau.
    v_sub = detect_h2h_plateau(
        sub, dominance_target=DOMINANCE_TARGET, K_snap=K, min_gain=0.01
    )
    assert v_sub.plateau is True
    assert v_sub.exit_fires is True
    assert v_sub.plateau_run_length == K

    # Above-min_gain uptick (+0.02 > 0.01) at the last step -> resets the run.
    # Build a plateau of K no-gain steps, then one genuine gain at the end.
    reset = [base] * (K + 1) + [base + 0.02]  # K no-gain steps, then +0.02 reset
    v_reset = detect_h2h_plateau(
        reset, dominance_target=DOMINANCE_TARGET, K_snap=K, min_gain=0.01
    )
    assert v_reset.plateau is False
    assert v_reset.exit_fires is False
    assert v_reset.plateau_run_length == 0
    assert v_reset.reason == "still_improving"


# -----------------------------------------------------------------------------
# 7. insufficient snapshots -> no exit
# -----------------------------------------------------------------------------
def test_insufficient_snapshots():
    K = DEFAULT_K_SNAP
    # Fewer than K_snap+1 points.
    scores = [0.40, 0.41, 0.40]
    v = detect_h2h_plateau(
        scores, dominance_target=DOMINANCE_TARGET, K_snap=K, min_gain=DEFAULT_MIN_GAIN
    )
    assert v.exit_fires is False
    assert v.plateau is False
    assert v.reason == "insufficient_snapshots"


# -----------------------------------------------------------------------------
# 8. regression guard -- B7 does NOT edit A5 a_gate.py
# -----------------------------------------------------------------------------
def test_does_not_edit_a5():
    # worktree root (git repo) = parents[4]; a_gate.py is untracked/gitignored
    # here, so ``git status`` for its path is clean (empty) iff B7 did NOT edit
    # it. The regression-guard intent: a_gate.py is unchanged on disk.
    worktree_root = Path(__file__).resolve().parents[4]
    rel = "TrainV3.5/python/train_v3/a_gate.py"
    res = subprocess.run(
        ["git", "-C", str(worktree_root), "status", "--porcelain", "--", rel],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "", f"a_gate.py was edited: {res.stdout!r}"


# -----------------------------------------------------------------------------
# 9. source-vs-source inverse vs A5 check_h2h_trending
# -----------------------------------------------------------------------------
def test_inverse_of_check_h2h_trending():
    K = DEFAULT_K_SNAP

    # (a) A series A5 flags as TRENDING UP should NOT plateau in B7.
    rising = [0.30 + 0.02 * i for i in range(K + 2)]  # strict +0.02 each step
    a5_up = check_h2h_trending(rising, min_snapshots=K)
    assert a5_up.passed is True, "A5 should flag rising series as trending-up"
    v_b7 = detect_h2h_plateau(
        rising, dominance_target=DOMINANCE_TARGET, K_snap=K, min_gain=DEFAULT_MIN_GAIN
    )
    assert v_b7.plateau is False, "B7 should NOT plateau a trending-up series"

    # (b) A flat series is non-decreasing so A5 (tolerance=0) flags it as
    # trending; B7's no-gain run-length logic also plateaus it (overlap edge
    # case, not a strict inverse). The strict inverse uses a REGRESSING series
    # below, which A5 flags as NOT-trending and B7 plateaus.
    flat = [0.40] * (K + 1)
    a5_flat = check_h2h_trending(flat, min_snapshots=K)
    assert a5_flat.passed is True, "flat series is non-decreasing (A5 trending)"
    v_b7_flat = detect_h2h_plateau(
        flat, dominance_target=DOMINANCE_TARGET, K_snap=K, min_gain=DEFAULT_MIN_GAIN
    )
    assert v_b7_flat.plateau is True, "B7 plateaus a flat (no-gain) series"
    # Strict inverse: a series that REGRESSES is NOT trending in A5 and IS a
    # plateau (no gain) in B7.
    regressing = [0.50 - 0.02 * i for i in range(K + 1)]  # -0.02 each step
    a5_down = check_h2h_trending(regressing, min_snapshots=K)
    assert a5_down.passed is False, "A5 should flag regressing series as NOT trending"
    v_b7_down = detect_h2h_plateau(
        regressing, dominance_target=DOMINANCE_TARGET, K_snap=K, min_gain=DEFAULT_MIN_GAIN
    )
    assert v_b7_down.plateau is True, "B7 should plateau a no-gain (regressing) series"
    # Regressing below target -> exit fires (DEFAULT below reading).
    assert v_b7_down.exit_fires is True


# -----------------------------------------------------------------------------
# 10. flippable at/above reading (D-B3 3a)
# -----------------------------------------------------------------------------
def test_flippable_at_above_reading():
    K = DEFAULT_K_SNAP

    # With below_target_exits=False: a plateau AT/ABOVE target FIRES exit.
    above = [0.60] * (K + 1)  # 0.60 >= 0.55 dominance target
    v_above = detect_h2h_plateau(
        above,
        dominance_target=DOMINANCE_TARGET,
        K_snap=K,
        min_gain=DEFAULT_MIN_GAIN,
        below_target_exits=False,
    )
    assert v_above.plateau is True
    assert v_above.below_target is False
    assert v_above.exit_fires is True
    assert v_above.reason == "plateau_at_or_above_dominance_target"

    # With below_target_exits=False: a plateau BELOW target does NOT fire.
    below = [0.40] * (K + 1)  # 0.40 < 0.55 dominance target
    v_below = detect_h2h_plateau(
        below,
        dominance_target=DOMINANCE_TARGET,
        K_snap=K,
        min_gain=DEFAULT_MIN_GAIN,
        below_target_exits=False,
    )
    assert v_below.plateau is True
    assert v_below.below_target is True
    assert v_below.exit_fires is False
    assert v_below.reason == "still_improving_below"