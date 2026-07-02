"""Block D component D3 -- ``c_to_d_handoff.py`` -- the C->D handoff module (NEW).

V5-Max pipeline position: Block C COMPLETE (C0-C4 done, C4 =
``c_loop_driver.py``) -> Block D (this file is D3, the C->D handoff that bridges
the C-loop exit manifest into a Block-D-READY seed pool + the E1 candidate set).
D3 sits BETWEEN the C-loop driver (C4) and the Block-D consolidation driver (D2):
the C-loop emits a ``CLoopManifest`` (C4 :186) carrying ``best_ever_path``; D3
turns that manifest into (a) a FRESH ``SnapshotPool`` seeded from the post-C
best-ever (D-D2) and (b) the threaded ``E1CandidateSet`` (post-C3 best, post-B
fallback, post-D filled at D2 exit) handed to E1 at the Block-D exit.

WHY NEW (``BLOCK_D_PLAN.md`` section 3 D3 + D-D2): the C-loop pool is the C-loop's
OWN transient pool (C4 owns its own ``SnapshotPool`` instance, fresh-built at
C-loop entry); Block D must NOT reuse the C-loop pool object (it carries C-loop
rolling snapshots that are NOT Block-D consolidation material). D-D2 = FRESH pool
seeded from post-C: ``build_block_d_seed_pool`` constructs a NEW ``SnapshotPool``
and sets the post-C best-ever as the immutable seed anchor + inaugural best-ever.
The post-B anchors (threaded from the C->D handoff caller) enter as ROLLING
non-anchors -- sparring partners for the short consolidation, NOT permanent
anchors (the pool non-anchor slot count is bounded; post-B peers are FIFO-eligible
sparring partners, not anchors -- INTENTIONAL per ``BLOCK_D_PLAN.md`` D3).

E1 CANDIDATES (``design.md:134``): E1 picks among three candidate checkpoints --
post-D (the Block-D exit checkpoint, filled by D2 at exit via
``E1CandidateSet.with_post_d``), post-C3 best (threaded from the C->D handoff as
``c_manifest.best_ever_path``), and post-B fallback (threaded from the caller as
``post_b_path``). ``thread_e1_candidates`` builds the set with post-D left None
(D2 fills it at Block-D exit).

CONSTRAINTS (frozen-classic guard): D3 is a NEW file. NO edit to ``c_loop_driver.py``
(C4 completed/verified) / ``snapshot_pool.py`` (B1 completed/verified) / any
``classic_*`` / ``reward_v5`` / ``v5_trace`` / ``core`` / ``state`` / ``league_v5``
/ ``gauntlet_v5`` / ``opponents_v5`` / ``rust_ffi`` / ``rust_ppo`` /
``rust_live_self_play`` / ``block_b_*`` / ``block_d_opponent_mix`` file. D3 imports
C4 ``CLoopManifest`` + B1 ``SnapshotPool`` / ``SnapshotEntry`` READ-ONLY. MLX/Rust
are NOT imported -- both collaborators are pure-python bookkeeping (no
MLX/Rust/ONNX), so the module + its tests are synthetic-testable without any
external dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .c_loop_driver import CLoopManifest
from .snapshot_pool import SnapshotEntry, SnapshotPool

__all__ = [
    "E1CandidateSet",
    "build_block_d_seed_pool",
    "thread_e1_candidates",
]


# =============================================================================
# E1 candidate set (``design.md:134`` -- E1 candidates: post-D, post-C3 best,
# post-B fallback). SINGLE-TYPED: the fields are ``str | None``; D2 passes the
# ``E1CandidateSet`` OBJECT (NOT raw strings) when filling post-D at Block-D exit.
# =============================================================================
@dataclass(frozen=True)
class E1CandidateSet:
    """The three E1 candidate checkpoint paths (``design.md:134``).

    ``post_d_path`` -- the Block-D exit checkpoint (filled by D2 at exit via
    ``with_post_d``; None until Block-D completes). ``post_c3_best_path`` -- the
    C-loop best-ever (threaded from ``CLoopManifest.best_ever_path`` at the
    C->D handoff). ``post_b_path`` -- the post-B fallback (threaded from the
    C->D handoff caller; None when no post-B anchor is supplied).

    FROZEN so the set is hashable + immutable between handoff points; the only
    mutation path is ``with_post_d``, which returns a NEW ``E1CandidateSet`` via
    ``dataclasses.replace`` (the original is never mutated -- D2 builds the
    post-D-filled copy and threads it to E1).
    """

    post_d_path: Optional[str] = None
    post_c3_best_path: Optional[str] = None
    post_b_path: Optional[str] = None

    def with_post_d(self, post_d_path: Optional[str]) -> "E1CandidateSet":
        """Return a NEW ``E1CandidateSet`` with ``post_d_path`` set (the original
        is unchanged -- frozen dataclass -> ``dataclasses.replace``). D2 calls
        this at Block-D exit to fill the post-D candidate before handing the set
        to E1."""
        return replace(self, post_d_path=post_d_path)


# =============================================================================
# build_block_d_seed_pool -- construct a FRESH ``SnapshotPool`` seeded from the
# post-C best-ever (D-D2) + optional post-B sparring partners (rolling non-
# anchors). Returns the pool (NOT a manifest) so the Block-D consolidation
# driver owns + mutates it directly.
# =============================================================================
def build_block_d_seed_pool(
    c_manifest: CLoopManifest,
    *,
    post_b_anchor_paths: Optional[list[str]] = None,
    post_c3_best_path: Optional[str] = None,
    h2h_vs_best: float = 0.0,
    p1_p2_gap: float = 0.0,
) -> SnapshotPool:
    """Build a FRESH ``SnapshotPool`` seeded from the post-C best-ever (D-D2).

    The post-C best-ever (``c_manifest.best_ever_path``) becomes the immutable
    seed anchor AND the inaugural best-ever (``SnapshotPool.set_seed_anchor`` is
    the single call that sets both -- ``snapshot_pool.py:213``: immutability
    guard at :218-223, SEED_ROLE reconstruction at :230, inaugural best-ever at
    :235-243). Post-B anchors (``post_b_anchor_paths``) enter as ROLLING non-
    anchors via ``add_snapshot`` (FIFO-eligible sparring partners for the short
    consolidation, NOT permanent anchors -- the pool non-anchor slot count is
    bounded; INTENTIONAL per ``BLOCK_D_PLAN.md`` D3).

    Args:
      c_manifest: the C-loop exit manifest (C4 ``CLoopManifest``). The
        ``best_ever_path`` field is the post-C best-ever checkpoint path used as
        the seed anchor + inaugural best-ever.
      post_b_anchor_paths: optional post-B anchor paths to add as rolling non-
        anchor sparring partners (default None -> no rolling entries).
      post_c3_best_path: reserved for the explicit post-C3 best path override
        (UNUSED here -- the seed anchor is ALWAYS ``c_manifest.best_ever_path``
        per D-D2; kept in the signature for forward-compat with a future
        manifest that carries a separate post-C3-best field). Defaults to None
        and does NOT affect the seed.
      h2h_vs_best: the H2H-vs-best score-rate to record on the seed anchor
        (default 0.0; the seed anchor is the inaugural best-ever so its
        ``h2h_vs_best`` is the floor that future Block-D snapshots must strictly
        beat -- ``snapshot_pool.py:324``).
      p1_p2_gap: the p1/p2 score-rate gap to record on the seed anchor (default
        0.0; threaded from the C-loop exit if the caller has it).

    Returns:
      A FRESH ``SnapshotPool`` with the post-C best-ever set as seed anchor +
      inaugural best-ever, plus any ``post_b_anchor_paths`` added as rolling
      non-anchors.

    Raises:
      ValueError: if ``c_manifest.best_ever_path`` is None -- the C-loop did not
        produce a checkpoint (C-loop skipped case), so Block D cannot seed its
        pool. Surfacing, NOT silent: a None best_ever_path means the C-loop
        never promoted a snapshot, and Block D has no anchor to consolidate
        from (``BLOCK_D_PLAN.md`` D3 + D-D2).
    """
    # D-D2: FRESH pool (NOT the C-loop pool object -- the C-loop pool is the
    # C-loop's own transient pool; Block D starts a new one seeded from post-C).
    pool = SnapshotPool()

    if c_manifest.best_ever_path is None:
        raise ValueError(
            "C-loop best_ever_path is None -- C-loop did not produce a "
            "checkpoint; cannot seed Block D pool"
        )

    # Post-C best-ever = immutable seed anchor + inaugural best-ever. NOTE:
    # ``SnapshotEntry.update_number`` is a REQUIRED field with NO default
    # (``snapshot_pool.py:82``); ``role`` is OMITTED -- ``set_seed_anchor``
    # reconstructs the entry with SEED_ROLE internally (``snapshot_pool.py:230``),
    # so passing role="seed" is a no-op that misreads the contract.
    pool.set_seed_anchor(
        SnapshotEntry(
            update_number=0,
            h2h_vs_best=float(h2h_vs_best),
            path=str(c_manifest.best_ever_path),
            p1_p2_gap=float(p1_p2_gap),
            promotion_eligible=True,
        )
    )

    # Post-B / seed anchors as peer rolling non-anchors (FIFO-eligible, NOT
    # anchors -- INTENTIONAL: the pool non-anchor slot count is bounded; post-B
    # peers are sparring partners for the short consolidation, not permanent
    # anchors). ``add_snapshot`` normalizes role to "rolling" internally
    # (``snapshot_pool.py:255-262``) and FIFO-evicts on overflow
    # (``snapshot_pool.py:267-274``); anchors are NEVER evicted.
    for p in (post_b_anchor_paths or []):
        pool.add_snapshot(
            SnapshotEntry(
                update_number=0,
                h2h_vs_best=0.0,
                path=str(p),
                p1_p2_gap=0.0,
                promotion_eligible=True,
                role="rolling",
            )
        )

    return pool


# =============================================================================
# thread_e1_candidates -- build the E1 candidate set from the C-loop manifest.
# post-C3 best = ``c_manifest.best_ever_path``; post-B = caller-supplied;
# post-D = None (filled by D2 at Block-D exit via ``E1CandidateSet.with_post_d``).
# =============================================================================
def thread_e1_candidates(
    c_manifest: CLoopManifest,
    post_b_path: Optional[str] = None,
) -> E1CandidateSet:
    """Build the ``E1CandidateSet`` threaded from the C->D handoff to E1.

    ``post_c3_best_path`` is ``c_manifest.best_ever_path`` (the C-loop best-
    ever, the post-C3 candidate E1 picks among). ``post_b_path`` is the caller-
    supplied post-B fallback (None when no post-B anchor is threaded). ``post_d_path``
    is None -- D2 fills it at Block-D exit via ``E1CandidateSet.with_post_d``
    (``design.md:134``: E1 candidates post-D / post-C3 best / post-B fallback).

    Args:
      c_manifest: the C-loop exit manifest (C4 ``CLoopManifest``). The
        ``best_ever_path`` field becomes ``post_c3_best_path``.
      post_b_path: optional post-B fallback checkpoint path (default None).

    Returns:
      An ``E1CandidateSet`` with ``post_c3_best_path=c_manifest.best_ever_path``,
      ``post_b_path=post_b_path``, ``post_d_path=None``.
    """
    return E1CandidateSet(
        post_c3_best_path=c_manifest.best_ever_path,
        post_b_path=post_b_path,
        post_d_path=None,
    )