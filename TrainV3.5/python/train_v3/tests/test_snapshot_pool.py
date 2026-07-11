"""B1 ``snapshot_pool.py`` tests (``BLOCK_B_PLAN.md:237-250``).

Synthetic only: fake checkpoints + ``_FakeCheckpointStore`` (no real MLX/Rust/ONNX).
``test_load_snapshot_as_self_prev_opponent`` wires a real A4 ``SelfPrevOpponent``
(``rust_live_self_play.py:279``) with a fake ``select_fn`` — a source-vs-source check
that the pool snapshot loads back into the A4 self-play opponent. The import is
skip-gated IF it ever pulls MLX/Rust (it does not in-worktree today, but the gate
future-proofs the test). The manifest round-trip test uses fake paths, NOT real npz
writes — no MLX required.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Callable

import numpy as np
import pytest

# Ensure the train_v3 package is importable when run from the worktree root via
# `python -m pytest` (PYTHONPATH is set by the runner; this is a belt-and-braces
# fallback so the file is robust to direct invocation from the tests/ dir).
_TV3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TV3 not in sys.path:
    sys.path.insert(0, _TV3)

from train_v3.snapshot_pool import (  # noqa: E402
    BEST_EVER_ROLE,
    ROLLING_ROLE,
    SEED_ROLE,
    CheckpointStore,
    SnapshotEntry,
    SnapshotPool,
)


# =============================================================================
# Fake checkpoint store (synthetic — no real npz / MLX)
# =============================================================================
class _FakeCheckpointStore:
    """``CheckpointStore`` impl backed by an in-memory dict of fake weights. Used by
    every test that needs to load a snapshot — production wires
    ``rust_trainer._save_checkpoint`` / ``model_mlx.save_checkpoint`` (read-only
    reuse, ``BLOCK_B_PLAN.md:213``)."""

    def __init__(self) -> None:
        self._paths: dict[str, dict[str, Any]] = {}

    def put(self, path: str, weights: dict[str, Any]) -> None:
        self._paths[path] = weights

    def exists(self, path: str) -> bool:
        return path in self._paths

    def load_weights(self, path: str) -> dict[str, Any]:
        if path not in self._paths:
            raise FileNotFoundError(f"fake checkpoint not found: {path}")
        return self._paths[path]


def _entry(update: int, h2h: float, path: str, *, gap: float = 0.0,
           eligible: bool = True, role: str = ROLLING_ROLE) -> SnapshotEntry:
    return SnapshotEntry(
        update_number=update,
        h2h_vs_best=h2h,
        path=path,
        p1_p2_gap=gap,
        promotion_eligible=eligible,
        role=role,
    )


# A4 SelfPrevOpponent import gate (skip the wiring test if the import chain ever
# pulls MLX/Rust). Verified importable without MLX in-worktree today.
def _import_self_prev_opponent():
    from train_v3.rust_live_self_play import SelfPrevOpponent, OpponentCtx
    return SelfPrevOpponent, OpponentCtx


def _self_prev_opponent_available() -> bool:
    try:
        _import_self_prev_opponent()
        return True
    except Exception:
        return False


# =============================================================================
# 1. FIFO eviction keeps anchors
# =============================================================================
def test_fifo_eviction_keeps_anchors():
    """Overflow evicts the OLDEST non-anchor; seed + best-ever retained
    (``BLOCK_B_PLAN.md:238``)."""
    pool = SnapshotPool(target_non_anchor_count=3)
    # Seed anchor (immutable) — also seeds the initial best-ever.
    seed = pool.set_seed_anchor(_entry(0, 0.55, "seed.npz", eligible=True))
    assert pool.seed_anchor is seed
    assert pool.best_ever is not None
    assert pool.best_ever.role == BEST_EVER_ROLE

    # Add 5 rolling snapshots into a target-3 pool → FIFO evicts the 2 oldest.
    for u in range(1, 6):
        pool.add_snapshot(_entry(u * 100, 0.40 + 0.01 * u, f"ckpt_{u}.npz"))

    rolling = pool.rolling
    assert len(rolling) == 3, f"expected 3 rolling after FIFO, got {len(rolling)}"
    # Oldest (u=1,2) evicted; u=3,4,5 retained in insertion order.
    assert [r.update_number for r in rolling] == [300, 400, 500]

    # Anchors NEVER evicted.
    assert pool.seed_anchor is not None
    assert pool.seed_anchor.update_number == 0
    assert pool.best_ever is not None
    # best-ever is the seed initially (no strict improvement offered by add_snapshot).
    assert pool.best_ever.update_number == 0
    # The seed/best-ever paths are NOT in the rolling evict set.
    rolling_paths = {r.path for r in rolling}
    assert "seed.npz" not in rolling_paths


# =============================================================================
# 2. Best-ever updates on strict improvement only (ties do NOT replace)
# =============================================================================
def test_best_ever_updates_on_strict_improvement():
    """Ties do NOT replace best-ever (mirrors A5 ``H2H_PROMOTION_THRESHOLD=0.5``
    strict beat at ``a_gate.py:657`` ``if not (h2h > thresh)``;
    ``BLOCK_B_PLAN.md:239``)."""
    pool = SnapshotPool()
    pool.set_seed_anchor(_entry(0, 0.55, "seed.npz"))
    assert pool.best_ever.h2h_vs_best == pytest.approx(0.55)

    # Tie at exactly the current best → NO replace.
    tie = _entry(10, 0.55, "tie.npz")
    replaced = pool.maybe_update_best_ever(tie)
    assert replaced is False
    assert pool.best_ever.update_number == 0
    assert pool.best_ever.path == "seed.npz"

    # Below the current best → NO replace.
    worse = _entry(11, 0.50, "worse.npz")
    assert pool.maybe_update_best_ever(worse) is False
    assert pool.best_ever.update_number == 0

    # Strict improvement (0.55 -> 0.61) → replace.
    better = _entry(12, 0.61, "better.npz")
    assert pool.maybe_update_best_ever(better) is True
    assert pool.best_ever.update_number == 12
    assert pool.best_ever.h2h_vs_best == pytest.approx(0.61)
    assert pool.best_ever.role == BEST_EVER_ROLE

    # Promotion-by-loss guard: maybe_update_best_ever consults ONLY external-bench
    # H2H. SnapshotEntry has NO ppo_loss / approx_kl / entropy fields — a regression
    # guard that internal metrics are structurally absent (inherited A5,
    # ``a_gate.py:637``).
    assert not hasattr(better, "ppo_loss")
    assert not hasattr(better, "approx_kl")
    assert not hasattr(better, "entropy")


# =============================================================================
# 3. Seed anchor immutable
# =============================================================================
def test_seed_anchor_immutable():
    """Second promotion does NOT overwrite the seed anchor
    (``BLOCK_B_PLAN.md:240``)."""
    pool = SnapshotPool()
    pool.set_seed_anchor(_entry(0, 0.55, "seed.npz"))
    assert pool.seed_anchor.path == "seed.npz"

    # A second set_seed_anchor must raise (immutable).
    with pytest.raises(RuntimeError, match="immutable"):
        pool.set_seed_anchor(_entry(7, 0.80, "seed2.npz"))

    # Seed unchanged.
    assert pool.seed_anchor.update_number == 0
    assert pool.seed_anchor.path == "seed.npz"
    assert pool.seed_anchor.role == SEED_ROLE

    # best-ever CAN still update (strict improvement) — only the SEED is frozen.
    pool.maybe_update_best_ever(_entry(8, 0.70, "better.npz"))
    assert pool.best_ever.update_number == 8
    assert pool.seed_anchor.update_number == 0  # still the original seed


# =============================================================================
# 4. Load snapshot as A4 SelfPrevOpponent — deterministic argmax
# =============================================================================
@pytest.mark.skipif(
    not _self_prev_opponent_available(),
    reason="A4 SelfPrevOpponent import unavailable (MLX/Rust not buildable in-worktree)",
)
def test_load_snapshot_as_self_prev_opponent():
    """A pool snapshot wired into A4 ``SelfPrevOpponent.select_fn`` yields a
    deterministic argmax (pure self-play when wired to a prior snapshot,
    ``BLOCK_B_PLAN.md:241``). Source-vs-source: real A4 ``SelfPrevOpponent``, fake
    weights + fake OpponentCtx."""
    SelfPrevOpponent, OpponentCtx = _import_self_prev_opponent()

    pool = SnapshotPool()
    pool.set_seed_anchor(_entry(0, 0.55, "seed.npz"))
    snap = pool.add_snapshot(_entry(200, 0.58, "snap_200.npz"))

    # Fake weights: a per-candidate score vector. The default factory picks the legal
    # action with the highest score (ties broken by lowest id — deterministic).
    scores = np.zeros(601, dtype=np.float32)
    scores[5] = 0.10
    scores[42] = 0.90  # the argmax
    scores[7] = 0.30
    store = _FakeCheckpointStore()
    store.put(snap.path, {"action_weights": scores})

    opp = pool.wire_self_prev_opponent(snap, store)
    assert isinstance(opp, SelfPrevOpponent)
    assert opp.name == "self"

    ctx = OpponentCtx(
        env_idx=0,
        actor_id=2,
        observation_v5=np.zeros(8, dtype=np.float32),
        legal_action_ids=np.array([5, 7, 42, 0], dtype=np.uintp),
        legal_action_features=None,
        legal_action_counts=4,
        mana_draw_legal=False,
    )
    a1 = opp.select(0, ctx)
    a2 = opp.select(0, ctx)
    # Deterministic: same (weights, ctx) -> same action_id.
    assert a1 == a2
    # Argmax: 42 has the highest score among the legal ids.
    assert a1 == 42

    # A DIFFERENT snapshot with different weights yields a different argmax on the
    # same ctx — confirms the select_fn is weights-dependent (not a constant stub).
    snap2 = pool.add_snapshot(_entry(400, 0.60, "snap_400.npz"))
    scores2 = np.zeros(601, dtype=np.float32)
    scores2[5] = 0.95  # now 5 is the argmax
    scores2[42] = 0.20
    store.put(snap2.path, {"action_weights": scores2})
    opp2 = pool.wire_self_prev_opponent(snap2, store)
    assert opp2.select(0, ctx) == 5


# =============================================================================
# 5. Manifest round-trip
# =============================================================================
def test_manifest_roundtrip():
    """Pool manifest writes + reloads with paths + metadata intact
    (``BLOCK_B_PLAN.md:242``)."""
    pool = SnapshotPool(target_non_anchor_count=4)
    pool.set_seed_anchor(_entry(0, 0.55, "seed.npz", gap=0.05, eligible=True))
    pool.add_snapshot(_entry(200, 0.58, "snap_200.npz", gap=0.08, eligible=True))
    pool.add_snapshot(_entry(400, 0.60, "snap_400.npz", gap=0.10, eligible=False))
    assert not pool.maybe_update_best_ever(_entry(400, 0.60, "snap_400.npz", gap=0.10, eligible=False))

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "pool_manifest.json")
        pool.write_manifest(path)

        # File exists + is JSON.
        assert os.path.exists(path)

        reloaded = SnapshotPool.read_manifest(path)

    # Config round-trips.
    assert reloaded.target_non_anchor_count == 4
    assert reloaded.best_ever_strict_threshold == pytest.approx(0.5)
    # The audited Block-B composition reserves 25% for self snapshots: V4-orig
    # is pressure, not the dominant frozen distribution.
    assert reloaded.frozen_non_self_share == pytest.approx(0.75)
    assert reloaded.prevalence_pool_target == 6

    # Seed anchor round-trips (paths + metadata intact).
    assert reloaded.seed_anchor is not None
    assert reloaded.seed_anchor.update_number == 0
    assert reloaded.seed_anchor.path == "seed.npz"
    assert reloaded.seed_anchor.h2h_vs_best == pytest.approx(0.55)
    assert reloaded.seed_anchor.p1_p2_gap == pytest.approx(0.05)
    assert reloaded.seed_anchor.promotion_eligible is True
    assert reloaded.seed_anchor.role == SEED_ROLE

    # Best-ever round-trips.
    assert reloaded.best_ever is not None
    assert reloaded.best_ever.update_number == 0
    assert reloaded.best_ever.path == "seed.npz"
    assert reloaded.best_ever.role == BEST_EVER_ROLE

    # Rolling round-trips (paths + metadata + order intact).
    rolling = reloaded.rolling
    assert [r.path for r in rolling] == ["snap_200.npz", "snap_400.npz"]
    assert [r.update_number for r in rolling] == [200, 400]
    assert rolling[1].promotion_eligible is False
    assert all(r.role == ROLLING_ROLE for r in rolling)

    # The reloaded pool is usable: add + FIFO still works, seed still immutable.
    reloaded.add_snapshot(_entry(600, 0.62, "snap_600.npz"))
    assert len(reloaded.rolling) == 3
    with pytest.raises(RuntimeError, match="immutable"):
        reloaded.set_seed_anchor(_entry(9, 0.99, "seed2.npz"))

    # to_manifest / from_manifest is a pure-dict round-trip too (no disk).
    again = SnapshotPool.from_manifest(pool.to_manifest())
    assert again.seed_anchor.path == "seed.npz"
    assert [r.path for r in again.rolling] == [r.path for r in pool.rolling]


def test_from_manifest_trims_overflow_rolling():
    """A hand-crafted / legacy manifest with more rolling entries than the target is
    trimmed by FIFO on reload — the reloaded pool is immediately well-formed and the
    anchors are NEVER evicted (defensive guard merged from the B1 adversarial review).
    """
    overfull_manifest = {
        "target_non_anchor_count": 3,
        "best_ever_strict_threshold": 0.5,
        "frozen_non_self_share": 0.75,
        "prevalence_pool_target": 6,
        "seed": _entry(0, 0.55, "seed.npz", eligible=True, role=SEED_ROLE).to_dict(),
        "best_ever": _entry(0, 0.55, "seed.npz", role=BEST_EVER_ROLE).to_dict(),
        # 5 rolling entries into a target-3 pool — should FIFO-trim to 3 on reload.
        "rolling": [
            _entry(u * 100, 0.40, f"legacy_{u}.npz").to_dict() for u in range(1, 6)
        ],
    }
    pool = SnapshotPool.from_manifest(overfull_manifest)
    # Trimmed to the target immediately (no add_snapshot needed to trigger eviction).
    assert len(pool.rolling) == 3
    # Oldest-first eviction: legacy_1, legacy_2 evicted; legacy_3,4,5 retained.
    assert [r.path for r in pool.rolling] == ["legacy_3.npz", "legacy_4.npz", "legacy_5.npz"]
    # Anchors NEVER evicted by the reload-time trim.
    assert pool.seed_anchor is not None
    assert pool.seed_anchor.path == "seed.npz"
    assert pool.best_ever is not None
    assert pool.best_ever.path == "seed.npz"
    # Pool is usable post-trim: a further add still FIFO-evicts correctly.
    pool.add_snapshot(_entry(600, 0.62, "snap_600.npz"))
    assert len(pool.rolling) == 3
    assert pool.rolling[0].path == "legacy_4.npz"


# =============================================================================
# 6. Pool grows self-snapshot prevalence (D-B5 hybrid support)
# =============================================================================
def test_pool_grows_self_snapshot_prevalence():
    """D-B5 hybrid (``BLOCK_B_PLAN.md:247-249`` + §2 DECISIONS CONFIRMED): the mix
    weight available to self-snapshots is a function of pool size — prevalence rises
    monotonically as the pool fills. V4-orig spectrum weights stay frozen."""
    pool = SnapshotPool(frozen_non_self_share=0.95, prevalence_pool_target=6)
    residual = 1.0 - 0.95  # 0.05 headroom for self-snapshots (spec-literal)

    # Empty pool → prevalence 0 (no snapshots to populate the share).
    assert pool.self_snapshot_prevalence_weight() == pytest.approx(0.0)

    weights = []
    for u in range(1, 8):  # fill past the target (6)
        pool.add_snapshot(_entry(u * 100, 0.40, f"snap_{u}.npz"))
        weights.append(pool.self_snapshot_prevalence_weight())

    # Monotone non-decreasing in pool size.
    assert all(weights[i] <= weights[i + 1] + 1e-9 for i in range(len(weights) - 1)), (
        f"prevalence must be monotone non-decreasing: {weights}"
    )

    # At target fill (6 non-anchors) prevalence == full residual. weights is 0-indexed
    # over u=1..7, so weights[5] corresponds to u=6 (6 non-anchors added, no FIFO yet).
    assert weights[5] == pytest.approx(residual)  # u=6 -> 6 non-anchors
    # Saturates at the residual past the target (u=7 -> FIFO caps at 6 non-anchors).
    assert weights[6] == pytest.approx(residual)  # u=7 -> 6 non-anchors (capped)
    assert weights[6] <= residual + 1e-9

    # The frozen non-self share is UNCHANGED by pool growth (V4-orig 0.40/0.20/0.15
    # frozen per D-B5). The collapse monitor that TRIGGERS reweighting lives in
    # B3/B4 — B1 only exposes the pool-size-driven weight.
    assert pool.frozen_non_self_share == pytest.approx(0.95)

    # Non-anchor count drives prevalence (anchors do NOT inflate it — a pool with
    # only a seed anchor and no rolling snapshots has prevalence 0).
    only_seed = SnapshotPool(frozen_non_self_share=0.95, prevalence_pool_target=6)
    only_seed.set_seed_anchor(_entry(0, 0.55, "seed.npz"))
    assert only_seed.self_snapshot_prevalence_weight() == pytest.approx(0.0)
