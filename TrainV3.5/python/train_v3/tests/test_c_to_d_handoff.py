"""Synthetic tests for the D3 C->D handoff module (``c_to_d_handoff.py``).

ALL collaborators are real-but-pure-python: the real ``CLoopManifest``
dataclass (C4 ``c_loop_driver.py:186`` -- a dataclass, NO MLX/Rust) + the real
``SnapshotPool`` / ``SnapshotEntry`` (B1 ``snapshot_pool.py`` -- pure
bookkeeping, NO MLX/Rust/ONNX). NO real MLX/Rust/ONNX/rlhf_env DB/socket is
touched -- D3 is pure bookkeeping that bridges the C-loop exit manifest into a
Block-D-READY seed pool + the E1 candidate set.

The tests assert SPECIFIC D3 behavior: post-C is the immutable seed anchor +
inaugural best-ever, post-B anchors are rolling non-anchors (NOT anchors), the
seed-anchor immutability guard fires on a second ``set_seed_anchor``, a None
``best_ever_path`` surfaces a ValueError (NOT silent), the E1 candidate set
carries post-C3 + post-B with post-D left None, and ``with_post_d`` returns a
NEW frozen ``E1CandidateSet``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import FrozenInstanceError

import pytest

# Ensure the train_v3 package is importable when run from the worktree root via
# `python -m pytest` (PYTHONPATH is set by the runner; this is a belt-and-braces
# fallback so the file is robust to direct invocation from the tests/ dir).
_TV3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TV3 not in sys.path:
    sys.path.insert(0, _TV3)

from train_v3.c_loop_driver import CLoopManifest  # noqa: E402
from train_v3.c_to_d_handoff import (  # noqa: E402
    E1CandidateSet,
    build_block_d_seed_pool,
    thread_e1_candidates,
)
from train_v3.snapshot_pool import (  # noqa: E402
    ROLLING_ROLE,
    SEED_ROLE,
    SnapshotEntry,
    SnapshotPool,
)


# =============================================================================
# build_block_d_seed_pool
# =============================================================================
def test_seed_pool_seeds_post_c_as_seed_anchor_and_best_ever():
    """D-D2: post-C best-ever is the immutable seed anchor AND the inaugural
    best-ever (``set_seed_anchor`` sets both, ``snapshot_pool.py:213``)."""
    manifest = CLoopManifest(best_ever_path="post_c.npz")
    pool = build_block_d_seed_pool(manifest)

    assert isinstance(pool, SnapshotPool)
    assert pool.seed_anchor is not None
    assert pool.seed_anchor.path == "post_c.npz"
    assert pool.seed_anchor.role == SEED_ROLE
    assert pool.best_ever is not None
    assert pool.best_ever.path == "post_c.npz"
    # The seed anchor + inaugural best-ever are the SAME checkpoint (post-C).
    assert pool.seed_anchor.path == pool.best_ever.path


def test_seed_pool_threads_h2h_and_p1p2_baseline_into_anchor_and_best_ever():
    """The ``h2h_vs_best`` / ``p1_p2_gap`` args thread into BOTH the seed anchor
    AND the inaugural best-ever (the post-C H2H baseline that future Block-D
    snapshots must STRICTLY beat, ``BLOCK_D_PLAN.md`` D3 + ``snapshot_pool.py``
    strict-beat floor). A regression that hardcoded 0.0 or swapped the fields
    would fail this (the default-args test above would not)."""
    manifest = CLoopManifest(best_ever_path="post_c.npz")
    pool = build_block_d_seed_pool(manifest, h2h_vs_best=0.7, p1_p2_gap=0.03)

    assert pool.seed_anchor.h2h_vs_best == pytest.approx(0.7)
    assert pool.seed_anchor.p1_p2_gap == pytest.approx(0.03)
    # The inaugural best-ever is the SAME checkpoint -> carries the same baseline.
    assert pool.best_ever.h2h_vs_best == pytest.approx(0.7)
    assert pool.best_ever.p1_p2_gap == pytest.approx(0.03)


def test_post_b_anchors_are_rolling_non_anchors():
    """Post-B anchors enter as ROLLING non-anchors (FIFO-eligible sparring
    partners, NOT permanent anchors). The seed anchor stays post-C."""
    manifest = CLoopManifest(best_ever_path="post_c.npz")
    pool = build_block_d_seed_pool(
        manifest, post_b_anchor_paths=["b1.npz", "b2.npz"]
    )

    # Seed anchor unchanged.
    assert pool.seed_anchor is not None
    assert pool.seed_anchor.path == "post_c.npz"

    # The two post-B entries are in the rolling pool (NOT anchors).
    rolling_paths = [e.path for e in pool.rolling]
    assert "b1.npz" in rolling_paths
    assert "b2.npz" in rolling_paths
    assert len(rolling_paths) == 2

    # Each rolling entry has role "rolling" (normalized by add_snapshot).
    for e in pool.rolling:
        assert e.role == ROLLING_ROLE

    # They are NOT anchors: anchors == {seed, best-ever} (both post-C here).
    anchor_paths = {a.path for a in pool.anchors}
    assert "b1.npz" not in anchor_paths
    assert "b2.npz" not in anchor_paths
    assert anchor_paths == {"post_c.npz"}

    # all_entries contains anchors first, then rolling.
    all_paths = [e.path for e in pool.all_entries]
    assert "post_c.npz" in all_paths
    assert "b1.npz" in all_paths
    assert "b2.npz" in all_paths


def test_seed_anchor_immutable_second_call_raises():
    """The seed anchor is immutable after first set (``snapshot_pool.py:218``).
    A second ``set_seed_anchor`` on the pool returned by
    ``build_block_d_seed_pool`` raises RuntimeError."""
    manifest = CLoopManifest(best_ever_path="post_c.npz")
    pool = build_block_d_seed_pool(manifest)

    # The pool already has a seed (post-C); a second set must raise.
    second = SnapshotEntry(
        update_number=1,
        h2h_vs_best=0.9,
        path="other.npz",
        p1_p2_gap=0.0,
        promotion_eligible=True,
    )
    with pytest.raises(RuntimeError):
        pool.set_seed_anchor(second)


def test_none_best_ever_path_raises():
    """A None ``best_ever_path`` surfaces a ValueError (C-loop skipped case --
    surfacing, NOT silent)."""
    manifest = CLoopManifest(best_ever_path=None)
    with pytest.raises(ValueError, match="best_ever_path is None"):
        build_block_d_seed_pool(manifest)


def test_build_block_d_seed_pool_default_no_post_b():
    """With ``post_b_anchor_paths=None`` the pool has ONLY the seed anchor (no
    rolling entries)."""
    manifest = CLoopManifest(best_ever_path="post_c.npz")
    pool = build_block_d_seed_pool(manifest)

    assert pool.seed_anchor is not None
    assert pool.seed_anchor.path == "post_c.npz"
    assert pool.best_ever is not None
    assert pool.best_ever.path == "post_c.npz"
    # No rolling entries.
    assert pool.rolling == ()
    assert pool.non_anchor_count == 0
    # Total pool size == 1 distinct anchor (seed == best-ever, both post-C).
    assert len(pool) == 1


# =============================================================================
# thread_e1_candidates + E1CandidateSet
# =============================================================================
def test_e1_candidate_set_carries_post_c3_and_post_b():
    """``thread_e1_candidates`` threads post-C3 best (= best_ever_path) + post-B
    fallback; post-D is None (filled by D2 at exit)."""
    manifest = CLoopManifest(best_ever_path="post_c3.npz")
    cand = thread_e1_candidates(manifest, post_b_path="post_b.npz")

    assert isinstance(cand, E1CandidateSet)
    assert cand.post_c3_best_path == "post_c3.npz"
    assert cand.post_b_path == "post_b.npz"
    assert cand.post_d_path is None


def test_e1_candidate_set_frozen_and_with_post_d():
    """``E1CandidateSet`` is frozen (field assignment raises
    ``FrozenInstanceError``); ``with_post_d`` returns a NEW set with post-D set
    and the other fields unchanged."""
    base = E1CandidateSet(
        post_c3_best_path="post_c3.npz",
        post_b_path="post_b.npz",
        post_d_path=None,
    )

    # Frozen: direct field assignment raises.
    with pytest.raises(FrozenInstanceError):
        base.post_d_path = "post_d.npz"  # type: ignore[misc]

    # with_post_d returns a NEW E1CandidateSet with post_d_path set; the
    # original is unchanged.
    filled = base.with_post_d("post_d.npz")
    assert isinstance(filled, E1CandidateSet)
    assert filled is not base
    assert filled.post_d_path == "post_d.npz"
    # The other fields are carried unchanged.
    assert filled.post_c3_best_path == "post_c3.npz"
    assert filled.post_b_path == "post_b.npz"

    # The original is NOT mutated by with_post_d.
    assert base.post_d_path is None
    assert base.post_c3_best_path == "post_c3.npz"
    assert base.post_b_path == "post_b.npz"


def test_thread_e1_candidates_default_no_post_b():
    """``thread_e1_candidates`` with no ``post_b_path`` leaves post-B None."""
    manifest = CLoopManifest(best_ever_path="post_c3.npz")
    cand = thread_e1_candidates(manifest)

    assert cand.post_c3_best_path == "post_c3.npz"
    assert cand.post_b_path is None
    assert cand.post_d_path is None


def test_e1_candidate_set_defaults_all_none():
    """The default ``E1CandidateSet()`` has all three fields None."""
    cand = E1CandidateSet()
    assert cand.post_d_path is None
    assert cand.post_c3_best_path is None
    assert cand.post_b_path is None


def test_with_post_d_on_default_set():
    """``with_post_d`` on a default (all-None) set only sets post-D."""
    base = E1CandidateSet()
    filled = base.with_post_d("post_d.npz")
    assert filled.post_d_path == "post_d.npz"
    assert filled.post_c3_best_path is None
    assert filled.post_b_path is None
    assert base.post_d_path is None