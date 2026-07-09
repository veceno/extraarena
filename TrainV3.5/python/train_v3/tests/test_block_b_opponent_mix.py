"""B3 ``block_b_opponent_mix.py`` tests (``BLOCK_B_PLAN.md:377-392`` + the workflow
B3 test spec).

Synthetic only: a fake pool (a tiny object exposing ``self_snapshot_prevalence_
weight()``) -- no real SnapshotPool / MLX / Rust / ONNX. The B1 prevalence-cap
regression guard (``test_b1_prevalence_cap_005``) uses a real ``SnapshotPool`` with
fake ``SnapshotEntry`` instances (no MLX -- just bookkeeping).
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
    BLOCK_B_IDENTITIES,
    BLOCK_B_TAIL_WEIGHTS,
    BLOCK_B_V4_ORIG_ALIASES,
    BLOCK_B_V4_ORIG_TOTAL,
    BLOCK_B_V4_ORIG_WEIGHTS,
    FROZEN_NON_SELF_TOTAL,
    SELF_SNAPSHOT_SHARE_CAP,
    build_block_b_mix_string,
    build_block_b_opponent_mix,
    collapse_reweight_boost,
    parse_block_b_opponent_mix,
    resolve_v4_orig_identity,
)
from train_v3.rust_live_self_play import (  # noqa: E402
    RULE_DISPATCH,
    resolve_opponent_dispatch,
)
from train_v3.snapshot_pool import SnapshotEntry, SnapshotPool  # noqa: E402
from train_v3.v4_orig_temp_spectrum import V4_ORIG_TEMP_ALIASES  # noqa: E402


# =============================================================================
# Fake pool (synthetic -- controlled self_snapshot_prevalence_weight, no MLX/Rust)
# =============================================================================
class _FakePool:
    """A tiny stand-in for B1 ``SnapshotPool`` exposing ONLY the B3 read-surface
    (``self_snapshot_prevalence_weight``). No real snapshots / MLX / Rust."""

    def __init__(self, prevalence: float) -> None:
        self._prevalence = float(prevalence)

    def self_snapshot_prevalence_weight(self) -> float:
        return self._prevalence


def _entry(update: int, path: str = "snap.npz") -> SnapshotEntry:
    return SnapshotEntry(
        update_number=update,
        h2h_vs_best=0.40,
        path=path,
        p1_p2_gap=0.0,
        promotion_eligible=True,
        role="rolling",
    )


# =============================================================================
# 1. test_mix_parses_with_aliases -- the full Block-B mix string validates via B3
#    parser, no unknown-name error (v4-orig-t07 / v4-orig-t12 accepted).
# =============================================================================
def test_mix_parses_with_aliases():
    """The full Block-B mix string (built by B3, including v4-orig-t07 /
    v4-orig-t12) validates via ``parse_block_b_opponent_mix`` -- no unknown-name
    error. The v4-orig-* names do NOT route through ``parse_v5_opponent_mix``
    (they are absent from ``V5_OPPONENT_KINDS``); B3's own validator accepts them
    directly via ``BLOCK_B_IDENTITIES``."""
    pool = _FakePool(SELF_SNAPSHOT_SHARE_CAP)  # full pool -> self-snapshot 0.05
    mix_str = build_block_b_mix_string(pool)
    parsed = parse_block_b_opponent_mix(mix_str)

    # Every parsed identity is canonical (in BLOCK_B_IDENTITIES).
    for name, _w in parsed:
        assert name in BLOCK_B_IDENTITIES, f"non-canonical identity: {name}"

    # The v4-orig-* names are present (the alias gap is closed -- they parse).
    parsed_names = {name for name, _ in parsed}
    assert "v4-orig-argmax" in parsed_names
    assert "v4-orig-t07" in parsed_names
    assert "v4-orig-t12" in parsed_names
    # punish_empty_board is in the mix and parses natively (no alias needed).
    assert "punish_empty_board" in parsed_names

    # A direct v4-orig-t07 string parses (the canonical-name gap is resolved).
    direct = parse_block_b_opponent_mix(
        "v4-orig-argmax:0.40,v4-orig-t07:0.20,v4-orig-t12:0.15"
    )
    assert {n for n, _ in direct} == {
        "v4-orig-argmax",
        "v4-orig-t07",
        "v4-orig-t12",
    }


def test_parse_rejects_unknown_name():
    """Mirror of ``parse_v5_opponent_mix``'s unknown-name rejection (oracle
    ``league_v5.py:57-58``)."""
    with pytest.raises(ValueError, match="unknown Block-B opponent kind"):
        parse_block_b_opponent_mix("not_a_real_identity:0.5")


def test_parse_skips_zero_and_empty_default_self():
    """Mirror ``parse_v5_opponent_mix``: skip weight<=0, empty -> self:1.0."""
    assert parse_block_b_opponent_mix("") == [("self", 1.0)]
    assert parse_block_b_opponent_mix("stall:0.0,end_turn:-0.1") == [("self", 1.0)]
    # Positive-weight entries kept; zero/negative skipped.
    parsed = parse_block_b_opponent_mix("stall:0.05,random:0.0,end_turn:0.01")
    assert parsed == [("stall", 0.05), ("end_turn", 0.01)]


# =============================================================================
# 2. test_punish_empty_board_dispatches_code5 -- resolves to Rust rule code 5 via
#    A4 resolve_opponent_dispatch after the :143 uncomment (EDIT 2).
# =============================================================================
def test_punish_empty_board_dispatches_code5():
    """After the additive uncomment of ``rust_live_self_play.py:143`` (EDIT 2,
    D-B10), ``resolve_opponent_dispatch('punish_empty_board')`` returns
    ``(RULE_DISPATCH, 5)`` -- the Rust ``ExploitAgentKind::PunishEmptyBoard`` code
    (``worker.rs:1258``). Zero Rust change; PunishEmptyBoard already exists."""
    kind, code = resolve_opponent_dispatch("punish_empty_board")
    assert kind == RULE_DISPATCH
    assert code == 5


# =============================================================================
# 3. test_weights_account_to_one -- self-snapshot residual + V4-orig + exploit +
#    tail = 1.0.
# =============================================================================
def test_weights_account_to_one():
    """The Block-B mix accounts to 1.0 at every pool fill level (D-B5 hybrid):
    self-snapshot residual + V4-orig 0.55/0.75 + exploit 0.15/0.75 + tail 0.05/0.75
    of the non-self budget."""
    for prevalence in (0.0, 0.01, 0.025, SELF_SNAPSHOT_SHARE_CAP):
        pool = _FakePool(prevalence)
        mix = build_block_b_opponent_mix(pool)
        total = sum(w for _, w in mix)
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"prevalence={prevalence}: mix sums to {total}, not 1.0"
        )

    # Group totals: self-snapshot == prevalence; non-self == 1 - prevalence split
    # across V4-orig / exploit / tail in the 0.55:0.15:0.05 of-0.75 ratio.
    pool = _FakePool(SELF_SNAPSHOT_SHARE_CAP)
    mix = dict(build_block_b_opponent_mix(pool))
    self_total = mix["self"] + mix["v5_snapshot"]
    assert self_total == pytest.approx(SELF_SNAPSHOT_SHARE_CAP)
    v4_total = mix["v4-orig-argmax"] + mix["v4-orig-t07"] + mix["v4-orig-t12"]
    exploit_total = (
        mix["stall"] + mix["anti_draw_greed"] + mix["punish_empty_board"]
    )
    tail_total = mix["greedy_face"] + mix["random"] + mix["end_turn"]
    non_self = 1.0 - SELF_SNAPSHOT_SHARE_CAP
    assert v4_total == pytest.approx(non_self * (BLOCK_B_V4_ORIG_TOTAL / FROZEN_NON_SELF_TOTAL))
    assert exploit_total == pytest.approx(non_self * (0.15 / FROZEN_NON_SELF_TOTAL))
    assert tail_total == pytest.approx(non_self * (0.05 / FROZEN_NON_SELF_TOTAL))


# =============================================================================
# 4. test_self_snapshot_prevalence_grows_with_pool -- larger pool -> larger
#    self-snapshot share; V4-orig RATIOS unchanged (D-B5 hybrid).
# =============================================================================
def test_self_snapshot_prevalence_grows_with_pool():
    """D-B5 hybrid: a larger pool yields a larger self-snapshot share; the V4-orig
    within-group RATIOS (0.40:0.20:0.15) are unchanged at every pool size."""
    small = _FakePool(0.05)
    large = _FakePool(SELF_SNAPSHOT_SHARE_CAP)
    small_mix = dict(build_block_b_opponent_mix(small))
    large_mix = dict(build_block_b_opponent_mix(large))

    small_self = small_mix["self"] + small_mix["v5_snapshot"]
    large_self = large_mix["self"] + large_mix["v5_snapshot"]
    assert large_self > small_self
    assert small_self == pytest.approx(0.05)
    assert large_self == pytest.approx(SELF_SNAPSHOT_SHARE_CAP)

    # V4-orig within-group RATIOS unchanged at both pool sizes.
    for mix in (small_mix, large_mix):
        a, t07, t12 = mix["v4-orig-argmax"], mix["v4-orig-t07"], mix["v4-orig-t12"]
        assert a / t07 == pytest.approx(0.40 / 0.20)
        assert a / t12 == pytest.approx(0.40 / 0.15)
        assert t07 / t12 == pytest.approx(0.20 / 0.15)


# =============================================================================
# 5. test_tail_reweighted_from_phase_a -- greedy_face 0.03 / random 0.01 /
#    end_turn 0.01 (NOT Phase-A 0.10 / 0.05 / 0.10).
# =============================================================================
def test_tail_reweighted_from_phase_a():
    """Block B reweights the tail from Phase-A 0.10 / 0.05 / 0.10 to 0.03 / 0.01 /
    0.01 (``BLOCK_B_PLAN.md:348-349``). The frozen within-group RATIOS are
    0.03 : 0.01 : 0.01 (3:1:1), and each tail weight is well below its Phase-A
    value."""
    # Frozen constants are the Block-B values, not Phase-A.
    assert BLOCK_B_TAIL_WEIGHTS == {
        "greedy_face": 0.03,
        "random": 0.01,
        "end_turn": 0.01,
    }

    pool = _FakePool(0.0)  # empty pool -> non_self budget = 1.0
    mix = dict(build_block_b_opponent_mix(pool))
    gf, rnd, et = mix["greedy_face"], mix["random"], mix["end_turn"]
    # Within-group ratio 3:1:1.
    assert gf / rnd == pytest.approx(0.03 / 0.01)
    assert rnd == pytest.approx(et)
    # Each tail weight is the frozen fraction of the non-self budget (<< Phase-A).
    non_self = 1.0
    assert gf == pytest.approx(non_self * (0.03 / FROZEN_NON_SELF_TOTAL))
    assert rnd == pytest.approx(non_self * (0.01 / FROZEN_NON_SELF_TOTAL))
    assert et == pytest.approx(non_self * (0.01 / FROZEN_NON_SELF_TOTAL))
    # NOT the Phase-A values (0.10 / 0.05 / 0.10) -- each is much smaller.
    assert gf < 0.05  # Phase-A greedy_face was 0.10
    assert rnd < 0.02  # Phase-A random was 0.05
    assert et < 0.02   # Phase-A end_turn was 0.10


# =============================================================================
# 6. test_frozen_ratios_preserved -- V4-orig 0.40:0.20:0.15 ratio preserved at any
#    pool size; exploit 0.05:0.05:0.05 equal; tail 0.03:0.01:0.01.
# =============================================================================
def test_frozen_ratios_preserved():
    """The frozen within-group RATIOS are preserved at every pool size (D-B5 hybrid
    grows the self-snapshot share but NEVER distorts the within-group RATIOS)."""
    for prevalence in (0.0, 0.02, 0.04, SELF_SNAPSHOT_SHARE_CAP):
        pool = _FakePool(prevalence)
        mix = dict(build_block_b_opponent_mix(pool))

        # V4-orig 0.40 : 0.20 : 0.15.
        a, t07, t12 = (
            mix["v4-orig-argmax"],
            mix["v4-orig-t07"],
            mix["v4-orig-t12"],
        )
        assert a / t07 == pytest.approx(0.40 / 0.20)
        assert a / t12 == pytest.approx(0.40 / 0.15)
        assert t07 / t12 == pytest.approx(0.20 / 0.15)

        # Exploit 0.05 : 0.05 : 0.05 (equal).
        s, adg, peb = mix["stall"], mix["anti_draw_greed"], mix["punish_empty_board"]
        assert s == pytest.approx(adg)
        assert adg == pytest.approx(peb)

        # Tail 0.03 : 0.01 : 0.01.
        gf, rnd, et = mix["greedy_face"], mix["random"], mix["end_turn"]
        assert gf / rnd == pytest.approx(0.03 / 0.01)
        assert rnd == pytest.approx(et)

        # The GROUP ratios are the frozen 0.55:0.15:0.05 of-0.75 at every pool size.
        v4_total = a + t07 + t12
        exploit_total = s + adg + peb
        tail_total = gf + rnd + et
        non_self = v4_total + exploit_total + tail_total
        assert v4_total / non_self == pytest.approx(BLOCK_B_V4_ORIG_TOTAL / FROZEN_NON_SELF_TOTAL)
        assert exploit_total / non_self == pytest.approx(0.15 / FROZEN_NON_SELF_TOTAL)
        assert tail_total / non_self == pytest.approx(0.05 / FROZEN_NON_SELF_TOTAL)


# =============================================================================
# 7. test_collapse_reweight_entry_point -- the boost function raises self-snapshot
    #    above 0.25 + compresses frozen non-self proportionally (D-B5 monitor hook).
# =============================================================================
def test_collapse_reweight_entry_point():
    """``collapse_reweight_boost(factor)`` is the mana_draw-collapse monitor ENTRY
    POINT (B3 exposes it; B4 wires the monitor logic). A factor > 1.0 RAISES the
    self-snapshot share above the 0.25 cap (compressing frozen non-self
    proportionally); the within-group frozen RATIOS are preserved; the mix still
    accounts to 1.0."""
    pool = _FakePool(SELF_SNAPSHOT_SHARE_CAP)  # full pool -> base self-snapshot 0.25

    # Baseline (no boost): self-snapshot == 0.25.
    base_mix = dict(build_block_b_opponent_mix(pool))
    base_self = base_mix["self"] + base_mix["v5_snapshot"]
    assert base_self == pytest.approx(SELF_SNAPSHOT_SHARE_CAP)

    # collapse_reweight_boost returns the reweight config (the entry point).
    cfg = collapse_reweight_boost(2.0)
    assert cfg == {"collapse_boost": 2.0}

    # Boosted: self-snapshot RAISED above 0.25 (to 0.50), non-self compressed.
    boosted_mix = dict(build_block_b_opponent_mix(pool, **cfg))
    boosted_self = boosted_mix["self"] + boosted_mix["v5_snapshot"]
    assert boosted_self > base_self
    assert boosted_self == pytest.approx(0.50)  # 0.25 * 2.0, below the 0.95 cap

    # Frozen non-self compressed proportionally (non_self = 0.50 vs base 0.75).
    non_self_boosted = 1.0 - boosted_self
    assert non_self_boosted == pytest.approx(0.50)

    # Within-group frozen RATIOS preserved under the boost.
    a, t07, t12 = (
        boosted_mix["v4-orig-argmax"],
        boosted_mix["v4-orig-t07"],
        boosted_mix["v4-orig-t12"],
    )
    assert a / t07 == pytest.approx(0.40 / 0.20)
    assert a / t12 == pytest.approx(0.40 / 0.15)
    s, adg, peb = (
        boosted_mix["stall"],
        boosted_mix["anti_draw_greed"],
        boosted_mix["punish_empty_board"],
    )
    assert s == pytest.approx(adg) and adg == pytest.approx(peb)

    # The group ratios are STILL the frozen 0.55:0.15:0.05 of-0.75 of non-self.
    v4_total = a + t07 + t12
    exploit_total = s + adg + peb
    tail_total = (
        boosted_mix["greedy_face"] + boosted_mix["random"] + boosted_mix["end_turn"]
    )
    assert v4_total / non_self_boosted == pytest.approx(BLOCK_B_V4_ORIG_TOTAL / FROZEN_NON_SELF_TOTAL)
    assert exploit_total / non_self_boosted == pytest.approx(
        0.15 / FROZEN_NON_SELF_TOTAL
    )
    assert tail_total / non_self_boosted == pytest.approx(
        0.05 / FROZEN_NON_SELF_TOTAL
    )

    # The mix still accounts to 1.0 under the boost.
    total = sum(w for _, w in build_block_b_opponent_mix(pool, **cfg))
    assert total == pytest.approx(1.0, abs=1e-9)

    # The boost is capped at _MAX_SELF_SHARE (0.95) -- a 0.05 non-self floor is kept.
    huge = build_block_b_opponent_mix(pool, **collapse_reweight_boost(100.0))
    huge_self = dict(huge)["self"] + dict(huge)["v5_snapshot"]
    assert huge_self <= 0.95 + 1e-9
    # And the non-self floor is never breached.
    assert sum(w for _, w in huge) == pytest.approx(1.0, abs=1e-9)

    # factor <= 0 is rejected.
    with pytest.raises(ValueError, match="collapse_boost"):
        collapse_reweight_boost(0.0)
    with pytest.raises(ValueError, match="collapse_boost"):
        build_block_b_opponent_mix(pool, collapse_boost=-1.0)


# =============================================================================
# 8. test_b1_prevalence_cap_025 -- B1's prevalence caps at 0.25 so self-snapshots
#    can become a real Block-B training lane.
# =============================================================================
def test_b1_prevalence_cap_025():
    """B1 ``SnapshotPool.self_snapshot_prevalence_weight`` caps at residual 0.25.
    The prevalence is 0 when the pool is empty, monotone-increasing as the pool
    fills, and saturates at 0.25 at/above ``prevalence_pool_target``."""
    # Default frozen_non_self_share leaves a 0.25 self-snapshot residual.
    pool = SnapshotPool(prevalence_pool_target=6)
    assert pool.frozen_non_self_share == pytest.approx(0.75)
    assert pool.self_snapshot_prevalence_weight() == pytest.approx(0.0)

    weights = []
    for u in range(1, 8):  # fill past the target (6)
        pool.add_snapshot(_entry(u * 100, f"snap_{u}.npz"))
        weights.append(pool.self_snapshot_prevalence_weight())

    # Monotone non-decreasing.
    assert all(weights[i] <= weights[i + 1] + 1e-9 for i in range(len(weights) - 1))
    # Saturates at the self-snapshot cap.
    assert weights[5] == pytest.approx(SELF_SNAPSHOT_SHARE_CAP)  # u=6 -> 6 non-anchors
    assert weights[6] == pytest.approx(SELF_SNAPSHOT_SHARE_CAP)  # u=7 -> capped
    assert weights[6] == pytest.approx(0.25)

    # B3 reads this prevalence: a full B1 pool feeds B3 a 0.25 self-snapshot share.
    mix = dict(build_block_b_opponent_mix(pool))
    self_total = mix["self"] + mix["v5_snapshot"]
    assert self_total == pytest.approx(SELF_SNAPSHOT_SHARE_CAP)
    assert sum(w for _, w in build_block_b_opponent_mix(pool)) == pytest.approx(
        1.0, abs=1e-9
    )


# =============================================================================
# 9. test_alias_map_covers_only_t07_t12 -- the B3 alias map resolves v4-orig-t07 /
#    v4-orig-t12 (and v4-orig-argmax) and does NOT alias punish_empty_board (it
#    parses natively); does NOT edit league_v5.V5_OPPONENT_KINDS.
# =============================================================================
def test_alias_map_covers_only_t07_t12():
    """The B3 alias map (``BLOCK_B_V4_ORIG_ALIASES``) covers ONLY the v4-orig-*
    names; ``punish_empty_board`` parses natively (no alias). The alias map lives in
    B3, NOT in ``league_v5.V5_OPPONENT_KINDS`` (frozen-classic READ-ONLY)."""
    # The alias map resolves the three v4-orig-* names to B2 identities.
    assert set(BLOCK_B_V4_ORIG_ALIASES) == {
        "v4-orig-argmax",
        "v4-orig-t07",
        "v4-orig-t12",
    }
    # It mirrors B2's V4_ORIG_TEMP_ALIASES (the alias layer re-exports B2's map).
    assert BLOCK_B_V4_ORIG_ALIASES == V4_ORIG_TEMP_ALIASES

    # resolve_v4_orig_identity returns the B2 identity name for each v4-orig-*.
    assert resolve_v4_orig_identity("v4-orig-argmax") == "v4-orig-argmax"
    assert resolve_v4_orig_identity("v4-orig-t07") == "v4-orig-t07"
    assert resolve_v4_orig_identity("v4-orig-t12") == "v4-orig-t12"

    # punish_empty_board is NOT in the alias map (it parses natively).
    assert "punish_empty_board" not in BLOCK_B_V4_ORIG_ALIASES
    with pytest.raises(KeyError):
        resolve_v4_orig_identity("punish_empty_board")

    # The alias map does NOT edit league_v5.V5_OPPONENT_KINDS: the v4-orig-* names
    # are STILL absent from V5_OPPONENT_KINDS (frozen-classic READ-ONLY).
    from train_v3.league_v5 import V5_OPPONENT_KINDS

    assert "v4-orig-t07" not in V5_OPPONENT_KINDS
    assert "v4-orig-t12" not in V5_OPPONENT_KINDS
    assert "v4-orig-argmax" not in V5_OPPONENT_KINDS
    # punish_empty_board IS in V5_OPPONENT_KINDS (via *EXPLOIT_AGENT_KINDS) -- it
    # parses natively, no alias needed.
    assert "punish_empty_board" in V5_OPPONENT_KINDS


# =============================================================================
# 10. test_mix_string_round_trips -- the built mix string re-parses to the same
#     weights (within float tolerance) and still accounts to 1.0.
# =============================================================================
def test_mix_string_round_trips():
    """``build_block_b_mix_string`` -> ``parse_block_b_opponent_mix`` preserves the
    weights (within float tolerance) and the accounting to 1.0 -- the string is
    fit for A4 ``sample_opponent_identities`` (``rust_live_self_play.py:471``)."""
    pool = _FakePool(0.03)
    mix = build_block_b_opponent_mix(pool)
    mix_str = build_block_b_mix_string(pool)
    reparsed = parse_block_b_opponent_mix(mix_str)

    # Same identity set (zero-weight entries are skipped by the parser).
    built_names = [n for n, w in mix if w > 0.0]
    reparsed_names = [n for n, _ in reparsed]
    assert built_names == reparsed_names

    # Weights round-trip within tolerance.
    built_by_name = {n: w for n, w in mix}
    for name, w in reparsed:
        assert w == pytest.approx(built_by_name[name], abs=1e-6)

    # Re-parsed mix accounts to 1.0.
    assert sum(w for _, w in reparsed) == pytest.approx(1.0, abs=1e-6)
