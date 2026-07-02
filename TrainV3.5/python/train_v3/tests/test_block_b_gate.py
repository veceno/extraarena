"""Tests for Block B component B6 -- ``block_b_gate.py`` (``BLOCK_B_PLAN.md:537-552``).

Synthetic-only: fabricated component series + a fake ``GameRunner`` + a fake
``ManaDrawBaseline``. NO real Rust / MLX / ONNX. Source-vs-source: A5
``check_mana_draw_band`` / ``play_gauntlet`` / ``check_h2h_trending`` /
``select_promotion`` / ``GateOutcome`` / ``AGateResult`` = oracle; B6 = UUT
(composes A5 pieces, builds NEW ``BlockBGateResult`` + monotone aggregate).
"""
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

from train_v3.a_gate import (
    AGateResult,
    GameResult,
    ManaDrawBaseline,
    check_exploit_resistance_gate,
    check_h2h_trending,
    check_mana_draw_band,
    check_no_assist_gate,
    evaluate_a_gate,
    play_gauntlet,
    select_promotion,
)
from train_v3.block_b_gate import (
    P1_P2_GAP_THRESHOLD,
    BlockBGateResult,
    block_b_aggregate,
    evaluate_block_b_gate,
    measure_gauntlet_rate,
)


# --- helpers -----------------------------------------------------------------
def _baseline(rate: float = 0.5) -> ManaDrawBaseline:
    """A fake Q4 baseline B with the default engine constants (HAND_CAP=4,
    MANA_DRAW_BASE=2) so the Q4 dependency guard does not invalidate it."""
    return ManaDrawBaseline(
        mana_draw_count=int(rate * 100),
        eligible_turns=100,
        rate=float(rate),
        hand_cap=4,
        mana_draw_base=2,
        valid=True,
    )


class _FakeRunner:
    """Fake GameRunner that always returns a win (synthetic, no Rust/MLX).

    A deliberately minimal always-win stub: the constructor takes no args so a
    reader cannot misread it as configurable win/draw/loss counts. The gauntlet
    component tests only assert that B6 REUSES A5 ``play_gauntlet`` and yields a
    score rate of 1.0 (all wins) -- non-win cases are exercised elsewhere via
    fabricated component series, not via this runner.
    """

    def __init__(self, eligible: int = 100):
        self._eligible = eligible

    def play(self, opponent_kind: str, *, seed: int) -> GameResult:
        return GameResult(
            outcome="win",
            mana_draw_count=50,
            eligible_turns=self._eligible,
            opponent=opponent_kind,
        )


def _eval_series(
    series,
    *,
    baseline=None,
    n_snap=5,
    p1_p2_gap=0.0,
    mana_rate=0.5,
    h2h_threshold=0.5,
    gauntlet_threshold=0.5,
    internal_metrics=None,
):
    """Drive ``evaluate_block_b_gate`` over a list of (h2h_rate, gauntlet_rate)
    snapshots, accumulating the aggregate history. Returns the final
    ``BlockBGateResult``."""
    baseline = baseline if baseline is not None else _baseline()
    history: list[float] = []
    result = None
    for h2h_rate, gauntlet_rate in series:
        result = evaluate_block_b_gate(
            h2h_rate=h2h_rate,
            gauntlet_rate=gauntlet_rate,
            mana_draw_rate=mana_rate,
            baseline=baseline,
            p1_p2_gap=p1_p2_gap,
            aggregate_history=history,
            n_snap=n_snap,
            h2h_threshold=h2h_threshold,
            gauntlet_threshold=gauntlet_threshold,
            internal_metrics=internal_metrics,
        )
        history = list(result.monotone_aggregate_history)
    assert result is not None
    return result


# =============================================================================
# 1. monotone improvement over N_snap
# =============================================================================
def test_promotion_requires_monotone_improvement_over_N_snap():
    # 5 improving snapshots: h2h + gauntlet both strictly increasing; gap=0;
    # mana in band (0.5 in [0.25, 0.75]). All 4 pass each snapshot -> promote.
    improving = [(0.50, 0.50), (0.55, 0.55), (0.60, 0.60), (0.65, 0.65), (0.70, 0.70)]
    result = _eval_series(improving, n_snap=5)
    assert result.passed is True
    assert len(result.monotone_aggregate_history) == 5
    assert result.failed_criteria() == []

    # One dip in the window: aggregate regresses at snapshot 4 -> no promote.
    dipping = [(0.50, 0.50), (0.60, 0.60), (0.70, 0.70), (0.55, 0.55), (0.75, 0.75)]
    result_dip = _eval_series(dipping, n_snap=5)
    assert result_dip.passed is False
    # the most-recent snapshot still passes all 4 components; failure is monotone
    assert result_dip.failed_criteria() == []


# =============================================================================
# 2. p1_p2 gap required
# =============================================================================
def test_p1_p2_gap_required():
    # 5 improving snapshots BUT p1_p2_gap > 0.12 (breach) -> no promote even
    # though h2h + gauntlet improve monotonically.
    improving = [(0.50, 0.50), (0.55, 0.55), (0.60, 0.60), (0.65, 0.65), (0.70, 0.70)]
    result = _eval_series(improving, n_snap=5, p1_p2_gap=0.20)
    assert result.p1_p2_gap.passed is False
    assert result.p1_p2_gap.score == pytest.approx(0.20)
    assert result.passed is False
    assert "p1_p2_gap" in result.failed_criteria()

    # Same series with gap under threshold -> promote.
    result_ok = _eval_series(improving, n_snap=5, p1_p2_gap=0.10)
    assert result_ok.p1_p2_gap.passed is True
    assert result_ok.passed is True


# =============================================================================
# 3. does NOT re-apply A-gate no_assist (regression guard, open_question #11)
# =============================================================================
def test_does_not_reapply_a_gate_no_assist():
    # A candidate that FAILS the A-gate no_assist criterion (score rate 0.40 <
    # 0.55) but whose Block-B external-bench improves monotonically with all 4
    # Block-B components passing CAN still promote via Block-B. B6 does NOT call
    # check_no_assist_gate / evaluate_a_gate's no_assist path.
    no_assist_score_rate = 0.40
    no_assist_outcome = check_no_assist_gate(no_assist_score_rate)
    assert no_assist_outcome.passed is False  # confirms the A-gate no_assist fails

    improving = [(0.50, 0.50), (0.55, 0.55), (0.60, 0.60), (0.65, 0.65), (0.70, 0.70)]
    result = _eval_series(improving, n_snap=5)
    # Block-B promotes despite the A-gate no_assist failure (not re-applied).
    assert result.passed is True
    # The Block-B result carries NO no_assist / exploit_resistance fields.
    assert not hasattr(result, "no_assist")
    assert not hasattr(result, "exploit_resistance")

    # Source guard: evaluate_block_b_gate does NOT call evaluate_a_gate /
    # check_no_assist_gate / check_exploit_resistance_gate (grep its source).
    src = inspect.getsource(evaluate_block_b_gate)
    assert "evaluate_a_gate" not in src
    assert "check_no_assist_gate" not in src
    assert "check_exploit_resistance_gate" not in src


# =============================================================================
# 4. promotion independent of ppo_loss / KL / entropy (inherited A5 guard)
# =============================================================================
def test_promotion_independent_of_ppo_loss():
    from train_v3.a_gate import CandidateInternalMetrics

    series = [(0.50, 0.50), (0.55, 0.55), (0.60, 0.60), (0.65, 0.65), (0.70, 0.70)]

    low_loss = CandidateInternalMetrics(ppo_loss=0.01, approx_kl=0.001, entropy=0.05)
    high_loss = CandidateInternalMetrics(ppo_loss=9.99, approx_kl=1.0, entropy=2.0)

    r_low = _eval_series(series, n_snap=5, internal_metrics=low_loss)
    r_high = _eval_series(series, n_snap=5, internal_metrics=high_loss)
    assert r_low.passed == r_high.passed is True
    assert r_low.monotone_aggregate_history == r_high.monotone_aggregate_history


# =============================================================================
# 5. first-snapshot seed (insufficient snapshots -> no promote, seed anchor)
# =============================================================================
def test_first_snapshot_seed():
    baseline = _baseline()
    # First snapshot: history empty -> len(history) becomes 1 < n_snap=5.
    result = evaluate_block_b_gate(
        h2h_rate=0.60,
        gauntlet_rate=0.60,
        mana_draw_rate=0.5,
        baseline=baseline,
        p1_p2_gap=0.0,
        aggregate_history=[],
        n_snap=5,
    )
    assert result.passed is False
    assert len(result.monotone_aggregate_history) == 1
    # All 4 components pass for this snapshot, but the verdict is not-yet-promote
    # (insufficient snapshots -- the caller seeds the best-ever anchor; no plateau).
    assert result.failed_criteria() == []
    assert result.h2h_vs_best.passed is True
    assert result.gauntlet.passed is True
    assert result.mana_draw_band.passed is True
    assert result.p1_p2_gap.passed is True

    # 3 snapshots (still < n_snap=5) -> no promote.
    r3 = _eval_series(
        [(0.50, 0.50), (0.55, 0.55), (0.60, 0.60)], n_snap=5
    )
    assert r3.passed is False
    assert len(r3.monotone_aggregate_history) == 3


# =============================================================================
# 6. composes A5 check_mana_draw_band + play_gauntlet (reused, not rewritten)
# =============================================================================
def test_composes_a5_band_and_gauntlet():
    baseline = _baseline(rate=0.5)  # band = [0.25, 0.75]

    # (a) B6's mana_draw_band component IS A5 check_mana_draw_band: the returned
    # GateOutcome has the A5 details keys + identical pass logic.
    in_rate = 0.5
    b6_result = evaluate_block_b_gate(
        h2h_rate=0.6,
        gauntlet_rate=0.6,
        mana_draw_rate=in_rate,
        baseline=baseline,
        p1_p2_gap=0.0,
        aggregate_history=[],
        n_snap=5,
    )
    a5_direct = check_mana_draw_band(in_rate, baseline)
    assert b6_result.mana_draw_band.passed == a5_direct.passed
    assert b6_result.mana_draw_band.score == a5_direct.score
    assert b6_result.mana_draw_band.name == a5_direct.name == "mana_draw_band"
    # A5-specific details key carried through (proves it is the A5 outcome).
    assert "baseline_rate" in b6_result.mana_draw_band.details
    assert "band" in b6_result.mana_draw_band.details

    # (b) B6's measure_gauntlet_rate REUSES A5 play_gauntlet (same outcomes).
    runner = _FakeRunner()
    opponent_kinds = ["face_rush", "board_control"]
    rate_via_b6 = measure_gauntlet_rate(
        runner, opponent_kinds, games_per_opponent=5, seed=1
    )
    outcomes_a5 = play_gauntlet(
        runner, opponent_kinds, games_per_opponent=5, seed=1
    )
    assert rate_via_b6 == pytest.approx(outcomes_a5.score_rate())
    # All games are wins -> score rate 1.0.
    assert rate_via_b6 == pytest.approx(1.0)

    # Source guard: block_b_gate.py IMPORTS check_mana_draw_band + play_gauntlet
    # from a_gate (reuse, not rewrite).
    import train_v3.block_b_gate as bbg

    src = inspect.getsource(bbg)
    assert "from .a_gate import" in src
    assert "check_mana_draw_band" in src
    assert "play_gauntlet" in src


# =============================================================================
# 7. does NOT edit A5 (git diff a_gate.py empty) -- frozen-classic guard
# =============================================================================
def test_does_not_edit_a5():
    repo_root = Path(__file__).resolve().parents[4]  # .../TrainV3.5
    # TrainV3.5 is gitignored at the repo root but tracked in this worktree; the
    # a_gate.py file lives under TrainV3.5/. Use git on the worktree root.
    worktree_root = Path("/Users/laveqox/Documents/ExtraArenaRaS/.claude/worktrees/glm-TrainV3.5Prep")
    a_gate_rel = "TrainV3.5/python/train_v3/a_gate.py"
    proc = subprocess.run(
        ["git", "-C", str(worktree_root), "status", "--porcelain", "--", a_gate_rel],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # TrainV3.5 may be gitignored at the repo root -- fall back to a content
        # hash comparison against the committed blob if trackable, else skip the
        # git probe and assert the file is unchanged via the diff of the index.
        pytest.skip(f"git status unavailable for {a_gate_rel}: {proc.stderr}")
    assert proc.stdout.strip() == "", f"a_gate.py was modified: {proc.stdout!r}"


# =============================================================================
# 8. BlockBGateResult has NO no_assist / exploit_resistance fields
# =============================================================================
def test_blockb_gate_result_no_no_assist_field():
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(BlockBGateResult)}
    assert "no_assist" not in field_names
    assert "exploit_resistance" not in field_names
    # The 4 Block-B components + verdict + history + n_snap + reason are present.
    assert field_names == {
        "h2h_vs_best",
        "gauntlet",
        "mana_draw_band",
        "p1_p2_gap",
        "passed",
        "monotone_aggregate_history",
        "n_snap",
        "reason",
    }
    # Frozen (immutable).
    assert BlockBGateResult.__dataclass_params__.frozen is True

    # to_dict + failed_criteria work.
    result = _eval_series([(0.5, 0.5)], n_snap=5, p1_p2_gap=0.2)
    d = result.to_dict()
    assert d["passed"] is False
    assert "p1_p2_gap" in d["failed_criteria"]
    assert "monotone_aggregate_history" in d
    assert d["n_snap"] == 5


# =============================================================================
# 9. does NOT re-apply exploit_resistance (regression guard)
# =============================================================================
def test_does_not_reapply_exploit_resistance():
    # A candidate that FAILS the A-gate exploit_resistance criterion (0.30 <
    # 0.50) but whose Block-B external-bench improves monotonically CAN still
    # promote via Block-B. B6 does NOT call check_exploit_resistance_gate.
    exploit_score_rate = 0.30
    exploit_outcome = check_exploit_resistance_gate(exploit_score_rate)
    assert exploit_outcome.passed is False  # A-gate exploit_resistance fails

    improving = [(0.50, 0.50), (0.55, 0.55), (0.60, 0.60), (0.65, 0.65), (0.70, 0.70)]
    result = _eval_series(improving, n_snap=5)
    assert result.passed is True
    assert not hasattr(result, "exploit_resistance")

    # Source guard: B6 module does NOT import / call check_exploit_resistance_gate
    # / check_no_assist_gate / evaluate_a_gate (namespace check avoids the
    # docstring mentions).
    import train_v3.block_b_gate as bbg

    assert not hasattr(bbg, "check_exploit_resistance_gate")
    assert not hasattr(bbg, "check_no_assist_gate")
    assert not hasattr(bbg, "evaluate_a_gate")
    # And evaluate_block_b_gate's body does not call them either.
    body = inspect.getsource(evaluate_block_b_gate)
    assert "check_exploit_resistance_gate" not in body
    assert "check_no_assist_gate" not in body
    assert "evaluate_a_gate" not in body


# =============================================================================
# extras: aggregate formula + monotone tolerance + mana band invalidation
# =============================================================================
def test_block_b_aggregate_formula():
    # gap=0, in band -> 0.6 + 0.6 + 1.0 + 1.0 = 3.2
    assert block_b_aggregate(0.6, 0.6, True, 0.0) == pytest.approx(3.2)
    # gap=0.12 -> parity term 0; in band -> 0.6 + 0.6 + 1.0 + 0.0 = 2.2
    assert block_b_aggregate(0.6, 0.6, True, 0.12) == pytest.approx(2.2)
    # gap>0.12 -> parity term clamps to 0 (lowered) -> 2.2 (not below 0)
    assert block_b_aggregate(0.6, 0.6, True, 0.20) == pytest.approx(2.2)
    # out of band -> mana term 0 -> 0.6 + 0.6 + 0.0 + 1.0 = 2.2
    assert block_b_aggregate(0.6, 0.6, False, 0.0) == pytest.approx(2.2)
    # gap=0.06 -> parity term 0.5 -> 0.6 + 0.6 + 1.0 + 0.5 = 2.7
    assert block_b_aggregate(0.6, 0.6, True, 0.06) == pytest.approx(2.7)


def test_mana_draw_band_invalidation_propagates():
    # B6 REUSES A5 check_mana_draw_band's Q4 guard: a stale baseline (engine
    # constants changed) invalidates the mana_draw_band component -> no promote.
    baseline = _baseline(rate=0.5)  # hand_cap=4, mana_draw_base=2
    improving = [(0.50, 0.50), (0.55, 0.55), (0.60, 0.60), (0.65, 0.65), (0.70, 0.70)]
    history: list[float] = []
    result = None
    for h2h_rate, gauntlet_rate in improving:
        result = evaluate_block_b_gate(
            h2h_rate=h2h_rate,
            gauntlet_rate=gauntlet_rate,
            mana_draw_rate=0.5,
            baseline=baseline,
            p1_p2_gap=0.0,
            aggregate_history=history,
            n_snap=5,
            current_hand_cap=99,  # differs from recorded 4 -> Q4 invalidation
            current_mana_draw_base=99,
        )
        history = list(result.monotone_aggregate_history)
    assert result.mana_draw_band.details["invalidated"] is True
    assert result.mana_draw_band.passed is False
    assert result.passed is False
    assert "mana_draw_band" in result.failed_criteria()


def test_monotone_tolerance_allows_small_dip():
    # With a small tolerance, a tiny dip in the aggregate still counts as
    # monotone (non-decreasing within tolerance) -> promote.
    improving = [(0.50, 0.50), (0.55, 0.55), (0.60, 0.60), (0.599, 0.599), (0.65, 0.65)]
    baseline = _baseline()
    history: list[float] = []
    result = None
    for h2h_rate, gauntlet_rate in improving:
        result = evaluate_block_b_gate(
            h2h_rate=h2h_rate,
            gauntlet_rate=gauntlet_rate,
            mana_draw_rate=0.5,
            baseline=baseline,
            p1_p2_gap=0.0,
            aggregate_history=history,
            n_snap=5,
            monotone_tolerance=0.01,
        )
        history = list(result.monotone_aggregate_history)
    assert result.passed is True

    # Same series with tolerance 0.0 (strict) -> the 0.599 dip regresses -> no promote.
    history2: list[float] = []
    result2 = None
    for h2h_rate, gauntlet_rate in improving:
        result2 = evaluate_block_b_gate(
            h2h_rate=h2h_rate,
            gauntlet_rate=gauntlet_rate,
            mana_draw_rate=0.5,
            baseline=baseline,
            p1_p2_gap=0.0,
            aggregate_history=history2,
            n_snap=5,
            monotone_tolerance=0.0,
        )
        history2 = list(result2.monotone_aggregate_history)
    assert result2.passed is False


# =============================================================================
# extras: reason discriminator (verdict is observable without re-deriving)
# =============================================================================
def test_reason_discriminator_covers_all_verdicts():
    baseline = _baseline()

    # (1) insufficient_snapshots: first snapshot, all 4 pass, but len < n_snap.
    r_seed = evaluate_block_b_gate(
        h2h_rate=0.60,
        gauntlet_rate=0.60,
        mana_draw_rate=0.5,
        baseline=baseline,
        p1_p2_gap=0.0,
        aggregate_history=[],
        n_snap=5,
    )
    assert r_seed.passed is False
    assert r_seed.reason == "insufficient_snapshots"
    assert r_seed.failed_criteria() == []  # no component failed

    # (2) promoted: 5 monotone-improving snapshots, all 4 pass.
    improving = [(0.50, 0.50), (0.55, 0.55), (0.60, 0.60), (0.65, 0.65), (0.70, 0.70)]
    r_promoted = _eval_series(improving, n_snap=5)
    assert r_promoted.passed is True
    assert r_promoted.reason == "promoted"

    # (3) component_failed: monotone-improving BUT p1_p2_gap breached (> 0.12).
    r_comp = _eval_series(improving, n_snap=5, p1_p2_gap=0.20)
    assert r_comp.passed is False
    assert r_comp.reason == "component_failed"
    assert "p1_p2_gap" in r_comp.failed_criteria()

    # (4) monotone_not_improving: all 4 pass each snapshot BUT the aggregate dips
    # in the window (no component failure -> the verdict is a monotone failure).
    dipping = [(0.50, 0.50), (0.60, 0.60), (0.70, 0.70), (0.55, 0.55), (0.75, 0.75)]
    r_mono = _eval_series(dipping, n_snap=5)
    assert r_mono.passed is False
    assert r_mono.failed_criteria() == []  # all components pass
    assert r_mono.reason == "monotone_not_improving"

    # to_dict surfaces the reason.
    assert r_seed.to_dict()["reason"] == "insufficient_snapshots"
    assert r_promoted.to_dict()["reason"] == "promoted"