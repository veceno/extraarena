"""D1 ``block_d_opponent_mix.py`` tests (``BLOCK_D_PLAN.md:57-63`` + the D1 test
spec).

Synthetic only: a fake pool (a tiny object exposing ``self_snapshot_prevalence_
weight()`` -- vestigial for D1, kept for API symmetry with B3) -- no real
SnapshotPool / MLX / Rust / ONNX. D1 uses ``self_share_target`` directly and does
NOT read ``pool.self_snapshot_prevalence_weight()``; the fake pool is passed only
to exercise the API surface.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure the train_v3 package is importable when run from the worktree root.
_TV3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TV3 not in sys.path:
    sys.path.insert(0, _TV3)

from train_v3.block_b_opponent_mix import (  # noqa: E402
    BLOCK_B_EXPLOIT_WEIGHTS,
    BLOCK_B_EXPLOIT_TOTAL,
    BLOCK_B_IDENTITIES,
    BLOCK_B_TAIL_WEIGHTS,
    BLOCK_B_TAIL_TOTAL,
    BLOCK_B_V4_ORIG_WEIGHTS,
    BLOCK_B_V4_ORIG_TOTAL,
    SELF_SNAPSHOT_SHARE_CAP,
    build_block_b_opponent_mix,
    collapse_reweight_boost as _b3_collapse_reweight_boost,
    parse_block_b_opponent_mix as _b3_parse_block_b_opponent_mix,
)
from train_v3.block_d_opponent_mix import (  # noqa: E402
    BLOCK_D_IDENTITIES,
    BLOCK_D_MAX_SELF_SHARE,
    build_block_d_mix_string,
    build_block_d_opponent_mix,
    collapse_reweight_boost,
    parse_block_d_opponent_mix,
)


# =============================================================================
# Fake pool (synthetic -- controlled self_snapshot_prevalence_weight, no MLX/Rust).
# Vestigial for D1 (D1 does NOT read prevalence); kept for API symmetry with B3.
# =============================================================================
class _FakePool:
    """A tiny stand-in for the pool exposing ONLY the (vestigial for D1) B3 read-
    surface (``self_snapshot_prevalence_weight``). D1 does NOT call it; the fake
    pool is passed only to exercise the D1 API surface (so D2 can swap builders).

    The prevalence saturates at ``SELF_SNAPSHOT_SHARE_CAP`` -- emulating B1
    ``SnapshotPool.self_snapshot_prevalence_weight`` saturation
    (``block_b_opponent_mix.py:156-157``, ``test_b1_prevalence_cap_005``). So a
    "full pool" (prevalence 1.0) feeds B3 a capped self-snapshot share (the B3 frozen-
    field cap), which is what the D-D1 consolidation-vs-frozen-field regression
    guard checks against. D1 ignores the value entirely."""

    def __init__(self, prevalence: float) -> None:
        self._prevalence = min(float(prevalence), SELF_SNAPSHOT_SHARE_CAP)

    def self_snapshot_prevalence_weight(self) -> float:
        return self._prevalence


def _complementary_shares(self_share_target: float) -> dict[str, float]:
    """Return non-self shares that sum with ``self_share_target`` to 1.0, preserving
    the D-D1 group ratio 0.30 : 0.15 : 0.05 (= 6 : 3 : 1) of the non-self budget.
    Used wherever a test varies ``self_share_target`` away from the 0.50 default so
    the builder's sum-to-1.0 assertion is satisfied."""
    non_self = 1.0 - float(self_share_target)
    return {
        "v4_orig_share": non_self * (0.30 / 0.50),
        "exploit_share": non_self * (0.15 / 0.50),
        "tail_share": non_self * (0.05 / 0.50),
    }


def _within_group_ratios(mix: dict[str, float]) -> None:
    """Assert the B3 frozen within-group RATIOS are preserved (0.40:0.20:0.15 V4-
    orig, 0.05:0.05:0.05 exploit, 0.03:0.01:0.01 tail)."""
    a, t07, t12 = mix["v4-orig-argmax"], mix["v4-orig-t07"], mix["v4-orig-t12"]
    assert a / t07 == pytest.approx(0.40 / 0.20)
    assert a / t12 == pytest.approx(0.40 / 0.15)
    assert t07 / t12 == pytest.approx(0.20 / 0.15)
    s, adg, peb = mix["stall"], mix["anti_draw_greed"], mix["punish_empty_board"]
    assert s == pytest.approx(adg)
    assert adg == pytest.approx(peb)
    gf, rnd, et = mix["greedy_face"], mix["random"], mix["end_turn"]
    assert gf / rnd == pytest.approx(0.03 / 0.01)
    assert rnd == pytest.approx(et)


# =============================================================================
# 1. test_mix_sums_to_one -- the mix accounts to 1.0 at every self_share_target
#    and (irrelevant) pool prevalence.
# =============================================================================
def test_mix_sums_to_one():
    """The Block-D mix accounts to 1.0 at ``self_share_target`` in {0.0, 0.25,
    0.50, 0.75, 1.0}, pool prevalence in {0.0, 0.05} (prevalence is NOT read by
    D1 -- the fake pool is vestigial), and ``collapse_boost`` in {0.5, 1.0, 2.0,
    4.0}. The 1.0 self_share_target case is the degenerate pure-self-play edge
    (all non-self shares 0): the cap is SKIPPED and ``self_snapshot_weight`` is
    forced to 1.0 (NOT ``self_share_target * collapse_boost``) so the mix sums to
    1.0 for EVERY collapse_boost -- the boost is meaningless with no non-self
    lanes to compress. This grid falsifies the breach where a naive
    ``self_share_target * collapse_boost`` degenerate branch would sum to
    collapse_boost != 1.0."""
    for prevalence in (0.0, 0.05):
        pool = _FakePool(prevalence)
        for self_share_target in (0.0, 0.25, 0.50, 0.75, 1.0):
            for collapse_boost in (0.5, 1.0, 2.0, 4.0):
                mix = build_block_d_opponent_mix(
                    pool,
                    self_share_target=self_share_target,
                    collapse_boost=collapse_boost,
                    **_complementary_shares(self_share_target),
                )
                total = sum(w for _, w in mix)
                assert total == pytest.approx(1.0, abs=1e-9), (
                    f"prevalence={prevalence}, self_share_target="
                    f"{self_share_target}, collapse_boost={collapse_boost}: "
                    f"mix sums to {total}, not 1.0"
                )


# =============================================================================
# 2. test_self_share_is_target_exact -- at collapse_boost=1.0 the self+v5_snapshot
#    weight == self_share_target EXACTLY (no B3-style 0->0.05 ramp); NEVER exceeds
#    self_share_target at boost=1.0.
# =============================================================================
def test_self_share_is_target_exact():
    """D1 uses ``self_share_target`` DIRECTLY (not ``pool.self_snapshot_prevalence_
    weight()``): at ``collapse_boost=1.0`` the self+v5_snapshot weight equals
    ``self_share_target`` EXACTLY, at every pool prevalence (the pool is not read).
    It NEVER exceeds ``self_share_target`` at boost=1.0 (the B3 ramp is gone)."""
    for prevalence in (0.0, 0.05):
        pool = _FakePool(prevalence)
        for self_share_target in (0.0, 0.25, 0.50, 0.75):
            mix = dict(build_block_d_opponent_mix(
                pool,
                self_share_target=self_share_target,
                **_complementary_shares(self_share_target),
            ))
            self_total = mix["self"] + mix["v5_snapshot"]
            assert self_total == pytest.approx(self_share_target, abs=1e-9), (
                f"prevalence={prevalence}, self_share_target={self_share_target}: "
                f"self+v5_snapshot={self_total}"
            )
            # NEVER exceeds self_share_target at boost=1.0.
            assert self_total <= self_share_target + 1e-12


# =============================================================================
# 2b. test_pool_independence_d1_does_not_read_prevalence -- D1 uses
#     self_share_target DIRECTLY; the mix is identical regardless of pool
#     prevalence. Falsifies an accidental prevalence read (prevalence 0.0 vs 1.0
#     would diverge if D1 consulted pool.self_snapshot_prevalence_weight()).
# =============================================================================
def test_pool_independence_d1_does_not_read_prevalence():
    """D1 must NOT read ``pool.self_snapshot_prevalence_weight()``: the mix at a
    full pool (prevalence 1.0, capped to 0.05 by ``_FakePool``) is BYTE-IDENTICAL
    to the mix at an empty pool (prevalence 0.0). A regression that wired
    prevalence into the base share would make these diverge (0.0 vs 0.05). This is
    a STATIC prevalence-independence check (not the cross-update byte-stability
    the plan warns against, ``BLOCK_D_PLAN.md:74``)."""
    mix_empty = build_block_d_opponent_mix(_FakePool(0.0))
    mix_full = build_block_d_opponent_mix(_FakePool(1.0))
    assert mix_empty == mix_full


# =============================================================================
# 3. test_group_share_assertion_fires -- a bad sum (1.01) raises ValueError.
# =============================================================================
def test_group_share_assertion_fires():
    """``build_block_d_opponent_mix`` raises ``ValueError`` when the four shares do
    not sum to 1.0 within 1e-6 (here 0.5+0.3+0.15+0.06 = 1.01)."""
    pool = _FakePool(0.05)
    with pytest.raises(ValueError, match="sum to 1.0"):
        build_block_d_opponent_mix(
            pool,
            self_share_target=0.5,
            v4_orig_share=0.3,
            exploit_share=0.15,
            tail_share=0.06,  # 0.5+0.3+0.15+0.06 = 1.01
        )


# =============================================================================
# 4. test_within_group_ratios_preserved -- the B3 frozen within-group RATIOS are
#    preserved at the default D-D1 shares.
# =============================================================================
def test_within_group_ratios_preserved():
    """At the default D-D1 shares (0.50/0.30/0.15/0.05) the within-group RATIOS are
    the B3 frozen ratios (V4-orig 0.40:0.20:0.15, exploit 0.05:0.05:0.05, tail
    0.03:0.01:0.01) -- D-D1 changes the GROUP shares, NOT the within-group ratios."""
    pool = _FakePool(0.05)
    mix = dict(build_block_d_opponent_mix(pool))  # defaults
    _within_group_ratios(mix)

    # Cross-check: each group's weight / its group total equals the B3 frozen ratio.
    a, t07, t12 = mix["v4-orig-argmax"], mix["v4-orig-t07"], mix["v4-orig-t12"]
    v4_total = a + t07 + t12
    assert a / v4_total == pytest.approx(
        BLOCK_B_V4_ORIG_WEIGHTS["v4-orig-argmax"] / BLOCK_B_V4_ORIG_TOTAL
    )
    assert t07 / v4_total == pytest.approx(
        BLOCK_B_V4_ORIG_WEIGHTS["v4-orig-t07"] / BLOCK_B_V4_ORIG_TOTAL
    )
    assert t12 / v4_total == pytest.approx(
        BLOCK_B_V4_ORIG_WEIGHTS["v4-orig-t12"] / BLOCK_B_V4_ORIG_TOTAL
    )

    s, adg, peb = mix["stall"], mix["anti_draw_greed"], mix["punish_empty_board"]
    exp_total = s + adg + peb
    for nm, w in (("stall", s), ("anti_draw_greed", adg), ("punish_empty_board", peb)):
        assert w / exp_total == pytest.approx(
            BLOCK_B_EXPLOIT_WEIGHTS[nm] / BLOCK_B_EXPLOIT_TOTAL
        )

    gf, rnd, et = mix["greedy_face"], mix["random"], mix["end_turn"]
    tail_total = gf + rnd + et
    for nm, w in (("greedy_face", gf), ("random", rnd), ("end_turn", et)):
        assert w / tail_total == pytest.approx(
            BLOCK_B_TAIL_WEIGHTS[nm] / BLOCK_B_TAIL_TOTAL
        )


# =============================================================================
# 5. test_collapse_boost_raises_self_capped_at_095 -- collapse_boost=4.0 raises
#    self+v5_snapshot to 0.95 (capped), mix still sums to 1.0, non-self compressed
#    proportionally.
# =============================================================================
def test_collapse_boost_raises_self_capped_at_095():
    """At ``self_share_target=0.50, collapse_boost=4.0`` the self+v5_snapshot weight
    is capped at ``BLOCK_D_MAX_SELF_SHARE`` (0.95 -- a 0.05 non-self floor is kept).
    The mix still sums to 1.0; the non-self groups compress proportionally; the
    within-group frozen RATIOS are preserved."""
    pool = _FakePool(0.05)
    mix = build_block_d_opponent_mix(
        pool, self_share_target=0.50, collapse_boost=4.0
    )
    d = dict(mix)
    self_total = d["self"] + d["v5_snapshot"]
    # 0.50 * 4.0 = 2.0 -> capped at 0.95.
    assert self_total == pytest.approx(BLOCK_D_MAX_SELF_SHARE, abs=1e-9)
    # Mix still sums to 1.0.
    assert sum(w for _, w in mix) == pytest.approx(1.0, abs=1e-9)
    # Non-self compressed proportionally (0.05 budget across the 3 groups).
    non_self = 1.0 - self_total
    assert non_self == pytest.approx(0.05, abs=1e-9)
    v4_total = d["v4-orig-argmax"] + d["v4-orig-t07"] + d["v4-orig-t12"]
    exp_total = d["stall"] + d["anti_draw_greed"] + d["punish_empty_board"]
    tail_total = d["greedy_face"] + d["random"] + d["end_turn"]
    # Group ratios are the D-D1 explicit shares (0.30:0.15:0.05 of 0.50) of non-self.
    assert v4_total / non_self == pytest.approx(0.30 / 0.50, abs=1e-9)
    assert exp_total / non_self == pytest.approx(0.15 / 0.50, abs=1e-9)
    assert tail_total / non_self == pytest.approx(0.05 / 0.50, abs=1e-9)
    # Within-group frozen RATIOS preserved under the boost.
    _within_group_ratios(d)


# =============================================================================
# 6. test_collapse_boost_zero_raises -- collapse_boost=0.0 raises ValueError.
# =============================================================================
def test_collapse_boost_zero_raises():
    """``collapse_boost <= 0`` is rejected (mirrors B3
    ``block_b_opponent_mix.py:298-299``)."""
    pool = _FakePool(0.05)
    with pytest.raises(ValueError, match="collapse_boost"):
        build_block_d_opponent_mix(pool, collapse_boost=0.0)
    with pytest.raises(ValueError, match="collapse_boost"):
        build_block_d_opponent_mix(pool, collapse_boost=-1.0)


# =============================================================================
# 7. test_parser_reexport_accepts_11_names_rejects_unknown -- the re-exported
#    parser accepts the 11 canonical names, rejects unknown, skips weight<=0.
# =============================================================================
def test_parser_reexport_accepts_11_names_rejects_unknown():
    """``parse_block_d_opponent_mix`` == ``parse_block_b_opponent_mix`` (re-exported,
    same 11-name set validates identically). Accepts the 11 canonical names, rejects
    unknown, skips ``weight <= 0`` -> default ``[("self", 1.0)]``."""
    # 4 entries parse (the names are all in BLOCK_D_IDENTITIES == BLOCK_B_IDENTITIES).
    parsed = parse_block_d_opponent_mix(
        "self:0.5,v4-orig-argmax:0.3,stall:0.15,greedy_face:0.05"
    )
    assert len(parsed) == 4
    assert {n for n, _ in parsed} == {
        "self", "v4-orig-argmax", "stall", "greedy_face"
    }

    # Unknown name rejected.
    with pytest.raises(ValueError, match="unknown Block-B opponent kind"):
        parse_block_d_opponent_mix("bogus:1.0")

    # weight<=0 skipped -> default self:1.0 (reaches the `out or [...]` fallback).
    assert parse_block_d_opponent_mix("self:0.0,random:0.0") == [("self", 1.0)]

    # Literal empty string -> default [("self", 1.0)]. This runs the
    # `(raw or "self:1.0")` short-circuit (``block_b_opponent_mix.py:240``) at line-
    # coverage level and would catch a parser that raised on empty input or returned
    # []. NOTE: it is NOT a branch discriminator -- the all-skipped fallback above
    # yields identical output (``"".split(",")`` -> ``[""]`` -> skip -> ``out=[]``
    # -> ``out or [("self",1.0)]``), so removing the short-circuit would still pass;
    # this is a coverage check, not a falsifier of the short-circuit branch.
    assert parse_block_d_opponent_mix("") == [("self", 1.0)]

    # The full 11-name set is accepted (round-trip the canonical identities).
    all_names = ",".join(f"{n}:0.05" for n in sorted(BLOCK_D_IDENTITIES))
    parsed_all = parse_block_d_opponent_mix(all_names)
    assert {n for n, _ in parsed_all} == set(BLOCK_D_IDENTITIES)
    # BLOCK_D_IDENTITIES == BLOCK_B_IDENTITIES (re-export, same 11-name set).
    assert BLOCK_D_IDENTITIES is BLOCK_B_IDENTITIES
    # parse_block_d_opponent_mix IS parse_block_b_opponent_mix (re-export, NOT a
    # shadowing re-implementation -- mirrors the collapse_reweight_boost identity
    # guard; a divergent copy mimicking the 4 tested behaviors but diverging on
    # untested edges would pass the behavioral suite, so the `is` guard is the
    # strong falsifier per the plan's "re-export rather than re-implement" directive).
    assert parse_block_d_opponent_mix is _b3_parse_block_b_opponent_mix


# =============================================================================
# 8. test_block_d_mix_not_block_b_frozen (REGRESSION GUARD) -- at a full fake pool
#    (prevalence 1.0), D1 default self share (0.50) != B3 self share (capped 0.25).
# =============================================================================
def test_block_d_mix_not_block_b_frozen():
    """THE D-D1 consolidation-vs-frozen-field guard. At a full fake pool (prevalence
    1.0), the D1 default self+v5_snapshot weight (0.50, ``self_share_target``
    DIRECT) is GREATER than B3's self+v5_snapshot weight (capped 0.25,
    ``pool.self_snapshot_prevalence_weight()`` ramp). This is the load-bearing gap
    D1 opens: a Block-B-frozen-weights verbatim run from the post-C checkpoint
    would NOT consolidate -- D1 does."""
    pool = _FakePool(1.0)
    d_mix = dict(build_block_d_opponent_mix(pool))  # D-D1 defaults
    b_mix = dict(build_block_b_opponent_mix(pool))  # B3 frozen
    d_self = d_mix["self"] + d_mix["v5_snapshot"]
    b_self = b_mix["self"] + b_mix["v5_snapshot"]
    assert d_self == pytest.approx(0.50, abs=1e-9)
    assert b_self == pytest.approx(SELF_SNAPSHOT_SHARE_CAP, abs=1e-9)
    assert d_self > b_self + 1e-6  # the consolidation-vs-frozen-field gap


# =============================================================================
# 9. test_mix_string_roundtrip -- build_block_d_mix_string -> parse -> sums ~1.0.
# =============================================================================
def test_mix_string_roundtrip():
    """``build_block_d_mix_string`` -> ``parse_block_d_opponent_mix`` round-trip:
    weights sum to ~1.0 within 1e-6 (high-precision rendering preserves the
    accounting). Zero-weight entries are emitted (parser skips them) so the identity
    set is stable."""
    pool = _FakePool(0.05)
    mix_str = build_block_d_mix_string(pool)
    reparsed = parse_block_d_opponent_mix(mix_str)
    assert sum(w for _, w in reparsed) == pytest.approx(1.0, abs=1e-6)

    # Round-trip at self_share_target=0.0 (zero-weight self/v5_snapshot emitted but
    # skipped by the parser -- the non-self identities still sum to 1.0).
    mix_str_zero = build_block_d_mix_string(
        pool, self_share_target=0.0, **_complementary_shares(0.0)
    )
    reparsed_zero = parse_block_d_opponent_mix(mix_str_zero)
    assert sum(w for _, w in reparsed_zero) == pytest.approx(1.0, abs=1e-6)
    # self / v5_snapshot skipped (zero weight).
    assert "self" not in {n for n, _ in reparsed_zero}
    assert "v5_snapshot" not in {n for n, _ in reparsed_zero}


# =============================================================================
# 10. test_v5_snapshot_always_present -- v5_snapshot is in the mix even at
#     self_share_target=0.0 (weight 0.0) -- identity set stable.
# =============================================================================
def test_v5_snapshot_always_present():
    """``_self_snapshot_split`` emits BOTH ``self`` and ``v5_snapshot`` (zero weight
    when ``self_snapshot_weight == 0``) so the mix identity set is stable at every
    ``self_share_target``. The parser skips zero-weight entries, but the BUILDER
    always emits both identities."""
    pool = _FakePool(0.05)
    mix = build_block_d_opponent_mix(
        pool, self_share_target=0.0, **_complementary_shares(0.0)
    )
    names = [n for n, _ in mix]
    assert "self" in names
    assert "v5_snapshot" in names
    # Both have zero weight at self_share_target=0.0.
    d = dict(mix)
    assert d["self"] == pytest.approx(0.0, abs=1e-12)
    assert d["v5_snapshot"] == pytest.approx(0.0, abs=1e-12)

    # At self_share_target=0.50 both are present with half the share each.
    mix_half = build_block_d_opponent_mix(pool, self_share_target=0.50)
    d_half = dict(mix_half)
    assert "self" in [n for n, _ in mix_half]
    assert "v5_snapshot" in [n for n, _ in mix_half]
    assert d_half["self"] == pytest.approx(0.25, abs=1e-9)
    assert d_half["v5_snapshot"] == pytest.approx(0.25, abs=1e-9)


# =============================================================================
# 11. test_collapse_reweight_boost_reexport -- the re-exported B3 entry point works
#     with the D1 builder (same collapse_boost semantics).
# =============================================================================
def test_collapse_reweight_boost_reexport():
    """``collapse_reweight_boost`` is re-exported from B3 (same entry-point
    semantics). ``build_block_d_opponent_mix(pool, **collapse_reweight_boost(f))``
    applies the boost identically to passing ``collapse_boost=f`` directly."""
    pool = _FakePool(0.05)
    cfg = collapse_reweight_boost(2.0)
    assert cfg == {"collapse_boost": 2.0}
    # The re-export IS the B3 function object (guards against a future shadowing
    # re-implementation -- a divergent copy would break D-D1/B3 collapse-boost
    # parity). This replaces a tautological `boosted == direct` check (same fn,
    # same args -> could not fail regardless of D1 correctness).
    assert collapse_reweight_boost is _b3_collapse_reweight_boost
    boosted = dict(build_block_d_opponent_mix(pool, **cfg))
    # 0.50 * 2.0 = 1.0 -> capped at BLOCK_D_MAX_SELF_SHARE (0.95).
    self_total = boosted["self"] + boosted["v5_snapshot"]
    assert self_total == pytest.approx(BLOCK_D_MAX_SELF_SHARE, abs=1e-9)

    # factor <= 0 rejected.
    with pytest.raises(ValueError, match="collapse_boost"):
        collapse_reweight_boost(0.0)
