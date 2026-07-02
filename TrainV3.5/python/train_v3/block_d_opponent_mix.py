"""Block D component D1 -- ``block_d_opponent_mix.py`` -- the consolidation
opponent-mix composition (D-D1, load-bearing).

V5-Max pipeline position: Block A in-worktree COMPLETE -> Block B (B1-B8) COMPLETE
-> Block C (C0-C4) COMPLETE -> Block D (D1-D3); this file is D1.

PURPOSE (``BLOCK_D_PLAN.md:57-63``): Block D is the post-C CONSOLIDATION -- "settle
post-C and prevent overfit to the last replay batch" (``design.md:130-131``). The
post-C learner is already strong (it cleared Block C), so it should settle against
its strong PEERS (self/peer-HEAVY prevalence) rather than grinding against the
frozen V4-orig field it already beats. B3's ``build_block_b_opponent_mix``
(``block_b_opponent_mix.py:276-325``) hard-codes a Block-B profile -- self+v5_snapshot
capped at 0.05, V4-orig 0.75, exploit 0.15, tail 0.05 -- a weak-learner-vs-frozen-
field league. Block D is the OPPOSITE: self/peer ~0.50, V4-orig ~0.30, exploit
~0.15, tail ~0.05 (D-D1 CONFIRMED, ``BLOCK_D_PLAN.md:49``). D1 builds that mix.

D-D1 DECISION CONFIRMED (user 2026-07-02, ``BLOCK_D_PLAN.md:49``): Consolidation
~0.50 -- self+v5_snapshot 0.50, V4-orig 0.30, exploit 0.15, tail 0.05. D-D4 = off
(no per-lane-loss reweight; ``BLOCK_D_PLAN.md:34``) -- NOT D1's concern; D1 just
builds the mix, D2 wires the no-op curriculum.

DISTINCT from B3 (``BLOCK_B_PLAN.md:326-346`` -- do NOT conflate B3's two gaps with
D1's): D1 does NOT touch the parse-side canonical set (``BLOCK_D_IDENTITIES`` IS
``BLOCK_B_IDENTITIES`` -- the SAME 11-name set) and does NOT touch the
``punish_empty_board`` dispatch-enable (A4 already uncommented for B3). D1's ONLY
load-bearing change vs B3 is the MIX SHAPE: explicit group shares (honoring D-D1's
"custom" option) with ``self_share_target`` applied DIRECTLY (NOT via
``pool.self_snapshot_prevalence_weight()`` -- that B1 method is the 0->0.05 ramp for
B3's weak-learner regime; Block D's pool is pre-seeded with post-C + post-B anchors
by D3, so the peer field is populated from update 1 and ``self_share_target``
applies as-is). The within-group frozen RATIOS are reused verbatim from B3 (READ-
ONLY constants).

CONSTRAINTS (frozen-classic guard, ``BLOCK_D_PLAN.md:88-94``): D1 is a NEW sibling
module. NO edit to ``classic_*/reward_v5/v5_trace/core/state/league_v5/gauntlet_v5/
opponents_v5/rust_ffi/rust_ppo/rust_live_self_play`` (all read-only). NO edit to
``block_b_opponent_mix.py`` (B3 is a completed, verified component -- re-verification
avoided); D1 imports B3 public constants + parse READ-ONLY. NO new dispatch
identities (the Block-D opponent set maps onto the existing 11 identities: ``self``
=post-C learner, ``v5_snapshot``=post-B/peer anchors, ``v4-orig-*``, exploit-lanes,
tail) -> NO A4 edit. Source-vs-source: B3 ``build_block_b_opponent_mix`` +
``parse_block_b_opponent_mix`` = oracle; D1 builder = UUT (mirror the within-group
frozen RATIOS, do NOT re-invent). Synthetic tests only (fake pool -- a tiny object;
no real MLX/Rust/ONNX).
"""

from __future__ import annotations

from typing import Protocol

# B3 frozen within-group weights + totals (READ-ONLY constants, ``block_b_opponent
# _mix.py:126-148``). Re-exported so D1 preserves the Block-B within-group RATIOS
# (0.40/0.20/0.15 V4-orig, 0.05/0.05/0.05 exploit, 0.03/0.01/0.01 tail) verbatim --
# D-D1 changes the GROUP shares, NOT the within-group ratios.
from .block_b_opponent_mix import (
    BLOCK_B_EXPLOIT_TOTAL,
    BLOCK_B_EXPLOIT_WEIGHTS,
    BLOCK_B_IDENTITIES,
    BLOCK_B_TAIL_TOTAL,
    BLOCK_B_TAIL_WEIGHTS,
    BLOCK_B_V4_ORIG_TOTAL,
    BLOCK_B_V4_ORIG_WEIGHTS,
    _self_snapshot_split,
    build_block_b_opponent_mix,  # convenience re-export for D2/test consumers (D1 does not call it internally)
    collapse_reweight_boost,
    parse_block_b_opponent_mix,
)

# =============================================================================
# Canonical identity set -- the SAME 11-name set as ``BLOCK_B_IDENTITIES``
# (``block_b_opponent_mix.py:105-117``). Re-exported under the D1 name for
# ownership clarity. NO new dispatch identities, NO A4 edit
# (``BLOCK_D_PLAN.md:58,61``).
# =============================================================================
BLOCK_D_IDENTITIES: frozenset[str] = BLOCK_B_IDENTITIES

#: Upper bound for the collapse-boosted self-snapshot share (mirrors B3
#: ``_MAX_SELF_SHARE`` = 0.95, ``block_b_opponent_mix.py:163``). The boost may
#: compress the non-self groups but never to zero -- a 0.05 non-self floor is kept
#: so the V4-orig + exploit + tail lanes always remain in the mix. Public on D1 so
#: tests can pin the cap without reaching into B3's private name.
BLOCK_D_MAX_SELF_SHARE: float = 0.95


# =============================================================================
# Pool protocol (the D1 surface -- vestigial; kept for API symmetry with B3 so D2
# can swap builders). D1 does NOT read ``pool.self_snapshot_prevalence_weight()``.
# =============================================================================
class _BlockDPool(Protocol):
    """The D1 read-surface on the pool -- VESTIGIAL.

    D1 uses ``self_share_target`` DIRECTLY (NOT ``pool.self_snapshot_prevalence_
    weight()`` -- that B1 method is the 0->0.05 ramp for B3's weak-learner regime;
    Block D's pool is pre-seeded with post-C + post-B anchors by D3, so the peer
    field is populated from update 1 and ``self_share_target`` applies as-is).
    The ``pool`` param is kept for API symmetry with B3 (so D2 can swap builders)
    and is RESERVED for a future B4-style collapse monitor (``BLOCK_D_PLAN.md:59``).
    The protocol declares ``self_snapshot_prevalence_weight`` for symmetry only --
    D1 NEVER calls it (``test_block_d_mix_not_block_b_frozen`` guards the
    consolidation-vs-frozen-field gap at a full pool).
    """

    def self_snapshot_prevalence_weight(self) -> float: ...


# =============================================================================
# Parser -- re-exported from B3 (the SAME 11-name set validates identically; do NOT
# re-implement, ``BLOCK_D_PLAN.md:60``).
# =============================================================================
#: ``parse_block_d_opponent_mix`` = ``parse_block_b_opponent_mix`` re-exported for
#: ownership clarity (``BLOCK_D_IDENTITIES`` == ``BLOCK_B_IDENTITIES`` so the
#: validator accepts/rejects identically; D2 feeds ``opponent_mix_parsed=mix``
#: directly to A4 on the runtime path, bypassing parse -- the validator is only
#: exercised in tests, ``BLOCK_D_PLAN.md:60``).
parse_block_d_opponent_mix = parse_block_b_opponent_mix


# =============================================================================
# Mix builder (D-D1 consolidation profile, always sums to 1.0)
# =============================================================================
def build_block_d_opponent_mix(
    pool: _BlockDPool,
    *,
    self_share_target: float = 0.50,
    v4_orig_share: float = 0.30,
    exploit_share: float = 0.15,
    tail_share: float = 0.05,
    collapse_boost: float = 1.0,
) -> list[tuple[str, float]]:
    """Build the Block-D consolidation opponent mix (D-D1, always sums to 1.0).

    Composition (D-D1 CONFIRMED, ``BLOCK_D_PLAN.md:49``):
      * ``self_snapshot_weight = min(self_share_target * collapse_boost,
        BLOCK_D_MAX_SELF_SHARE)`` -- ``self_share_target`` is used DIRECTLY (NOT via
        ``pool.self_snapshot_prevalence_weight()``; see the ``_BlockDPool`` docstring
        for why). ``collapse_boost > 1.0`` (the mana_draw-collapse monitor, B2/B8
        hook) RAISES the self-snapshot share above ``self_share_target`` (compressing
        the non-self groups proportionally in the boost direction), capped at
        ``BLOCK_D_MAX_SELF_SHARE`` (0.95 -- a hard ceiling: a 0.05 non-self floor is
        kept so the V4-orig + exploit + tail lanes always remain). The cap is a
        ceiling on self, so it compresses non-self when ``self_share_target *
        collapse_boost > 0.95``; in the nonsensical case ``self_share_target > 0.95``
        with non-self > 0 (outside the D-D1 operating range -- default 0.50) the cap
        instead pushes non-self UP above its target shares, but the mix still sums to
        1.0. The cap is SKIPPED when ``non_self_target_total == 0.0``
        (the degenerate pure-self-play case ``self_share_target == 1.0`` with all
        non-self shares 0) -- the boost is meaningless with no non-self lanes to
        compress, so ``self_snapshot_weight = 1.0`` (NOT
        ``self_share_target * collapse_boost``) and the mix sums to 1.0 for every
        ``collapse_boost``. Clamp non-negative.
      * ``non_self_budget = 1 - self_snapshot_weight``, distributed across the 3
        groups by RELATIVE shares: ``v4_orig_budget = non_self_budget *
        (v4_orig_share / non_self_target_total)`` where ``non_self_target_total =
        v4_orig_share + exploit_share + tail_share``; likewise exploit/tail. When
        ``collapse_boost`` raised self above target, ``non_self_budget <
        non_self_target_total``, so the groups compress proportionally -- the
        intended behaviour.
      * WITHIN each group the B3 frozen within-group RATIOS are preserved verbatim
        (V4-orig 0.40 : 0.20 : 0.15, exploit 0.05 : 0.05 : 0.05, tail 0.03 : 0.01 :
        0.01) by reusing ``BLOCK_B_*_WEIGHTS`` / ``BLOCK_B_*_TOTAL``.
      * Self-snapshot split: ``_self_snapshot_split(self_snapshot_weight)`` (B3
        ``block_b_opponent_mix.py:261-273``) -- emits ``[("self", half),
        ("v5_snapshot", half)]`` so BOTH identities are always present (zero weight
        when ``self_snapshot_weight == 0``). Emitted FIRST.

    The ``pool`` param is VESTIGIAL -- D1 does NOT read
    ``pool.self_snapshot_prevalence_weight()`` (the Block D pool is pre-seeded by
    D3, peer field populated from update 1; ``self_share_target`` applies as-is,
    ``BLOCK_D_PLAN.md:59``). Kept for API symmetry with B3 so D2 can swap builders;
    RESERVED for a future B4-style collapse monitor.

    The mix ALWAYS sums to 1.0 (``test_mix_sums_to_one``). The frozen within-group
    RATIOS are preserved at any group shares (``test_within_group_ratios_preserved``).

    Raises ``ValueError`` when the four shares do not sum to 1.0 within 1e-6 or when
    ``collapse_boost <= 0``.
    """
    if abs(
        self_share_target + v4_orig_share + exploit_share + tail_share - 1.0
    ) > 1e-6:
        raise ValueError(
            "Block-D group shares must sum to 1.0 (within 1e-6): got "
            f"self={self_share_target}, v4_orig={v4_orig_share}, "
            f"exploit={exploit_share}, tail={tail_share}"
        )
    if collapse_boost <= 0.0:
        raise ValueError("collapse_boost must be > 0")

    non_self_target_total = (
        float(v4_orig_share) + float(exploit_share) + float(tail_share)
    )
    # The 0.95 self cap keeps a 0.05 non-self floor so the V4-orig + exploit + tail
    # lanes always remain. The degenerate pure-self-play case
    # (non_self_target_total == 0.0, i.e. self_share_target == 1.0 with all non-self
    # shares 0) is handled in the else-branch below (force self_snapshot_weight = 1.0
    # -- see that branch for the rationale).
    if non_self_target_total > 0.0:
        self_snapshot_weight = min(
            float(self_share_target) * float(collapse_boost), BLOCK_D_MAX_SELF_SHARE
        )
    else:
        # Degenerate pure-self-play case (non_self_target_total == 0.0, i.e.
        # self_share_target == 1.0 with all non-self shares 0): the boost is
        # MEANINGLESS here -- there are no non-self lanes to compress, so the
        # boosted residual has nowhere to go. Force self_snapshot_weight = 1.0
        # (NOT self_share_target * collapse_boost, which would sum to collapse_boost
        # != 1.0 and breach the "ALWAYS sums to 1.0" contract for every
        # collapse_boost != 1.0). The 0.05 non-self floor is vacuous with no
        # non-self lanes. This is outside the D-D1 operating range (default
        # 0.50/0.30/0.15/0.05, where non_self_target_total == 0.50 > 0 and the cap
        # + proportional compression work correctly), but keeps the contract true
        # for every accepted input (``test_mix_sums_to_one`` covers
        # self_share_target in {0,0.25,0.5,0.75,1.0} x collapse_boost in
        # {0.5,1.0,2.0,4.0}).
        self_snapshot_weight = 1.0
    # Clamp non-negative (a negative self_share_target is nonsensical; guard anyway).
    if self_snapshot_weight < 0.0:
        self_snapshot_weight = 0.0

    non_self_budget = 1.0 - self_snapshot_weight

    mix: list[tuple[str, float]] = []
    # Self-snapshot split FIRST (both identities always present; zero weight when
    # self_snapshot_weight == 0).
    mix.extend(_self_snapshot_split(self_snapshot_weight))

    # V4-orig group (within-group 0.40 : 0.20 : 0.15 ratio preserved verbatim).
    if non_self_target_total > 0.0:
        v4_orig_budget = non_self_budget * (
            float(v4_orig_share) / non_self_target_total
        )
    else:
        v4_orig_budget = 0.0
    for name, frozen_w in BLOCK_B_V4_ORIG_WEIGHTS.items():
        mix.append((name, v4_orig_budget * (frozen_w / BLOCK_B_V4_ORIG_TOTAL)))

    # Exploit group (within-group 0.05 : 0.05 : 0.05 equal).
    if non_self_target_total > 0.0:
        exploit_budget = non_self_budget * (
            float(exploit_share) / non_self_target_total
        )
    else:
        exploit_budget = 0.0
    for name, frozen_w in BLOCK_B_EXPLOIT_WEIGHTS.items():
        mix.append((name, exploit_budget * (frozen_w / BLOCK_B_EXPLOIT_TOTAL)))

    # Tail group (within-group 0.03 : 0.01 : 0.01).
    if non_self_target_total > 0.0:
        tail_budget = non_self_budget * (
            float(tail_share) / non_self_target_total
        )
    else:
        tail_budget = 0.0
    for name, frozen_w in BLOCK_B_TAIL_WEIGHTS.items():
        mix.append((name, tail_budget * (frozen_w / BLOCK_B_TAIL_TOTAL)))

    return mix


def build_block_d_mix_string(
    pool: _BlockDPool,
    *,
    self_share_target: float = 0.50,
    v4_orig_share: float = 0.30,
    exploit_share: float = 0.15,
    tail_share: float = 0.05,
    collapse_boost: float = 1.0,
) -> str:
    """Render the Block-D mix as a parseable mix string (for validation via
    ``parse_block_d_opponent_mix`` + for A4 ``sample_opponent_identities``
    ``rust_live_self_play.py:471``).

    Mirrors B3 ``build_block_b_mix_string`` (``block_b_opponent_mix.py:349-362``):
    the string is ``"name:weight,name:weight,..."`` with high-precision weights so
    the round-trip through ``parse_block_d_opponent_mix`` preserves the accounting
    to 1.0 (within float tolerance). Zero-weight entries are emitted (the parser
    skips them) so the identity set is stable.
    """
    mix = build_block_d_opponent_mix(
        pool,
        self_share_target=self_share_target,
        v4_orig_share=v4_orig_share,
        exploit_share=exploit_share,
        tail_share=tail_share,
        collapse_boost=collapse_boost,
    )
    return ",".join(f"{name}:{weight:.10f}" for name, weight in mix)


__all__ = [
    "BLOCK_B_EXPLOIT_TOTAL",
    "BLOCK_B_EXPLOIT_WEIGHTS",
    "BLOCK_B_IDENTITIES",
    "BLOCK_B_TAIL_TOTAL",
    "BLOCK_B_TAIL_WEIGHTS",
    "BLOCK_B_V4_ORIG_TOTAL",
    "BLOCK_B_V4_ORIG_WEIGHTS",
    "BLOCK_D_IDENTITIES",
    "BLOCK_D_MAX_SELF_SHARE",
    "build_block_b_opponent_mix",
    "build_block_d_mix_string",
    "build_block_d_opponent_mix",
    "collapse_reweight_boost",
    "parse_block_b_opponent_mix",
    "parse_block_d_opponent_mix",
]