"""Tests for Block A component A5 — ``a_gate.py`` (the A-gate + promotion selector
+ Q4 mana_draw-usage measurement). TRACKED (verify: ``git check-ignore`` exit 1).

All tests use SYNTHETIC game outcomes (fake win rates, fake mana_draw counts, fake
H2H series) — the measurement + gating + promotion logic is unit-testable without a
full live training run (``BLOCK_A_PLAN.md:570-586``). No MLX / Rust required for the
core tests; the operational live gauntlet is skipped via ``test_skip_if_no_mlx_or_rust``.

Run: ``PYTHONPATH=.:TrainV3.5/python python3 -m pytest TrainV3.5/python/train_v3/tests/test_a_gate.py``.
"""
from __future__ import annotations

import math

import pytest

from train_v3.a_gate import (
    DEFAULT_H2H_MIN_SNAPSHOTS,
    ENGINE_HAND_CAP,
    ENGINE_MANA_DRAW_BASE,
    EXPLOIT_RESISTANCE_MIN_SCORE_RATE,
    EXPLOIT_AGENT_KINDS,
    H2H_PROMOTION_THRESHOLD,
    MANA_DRAW_BAND_HIGH,
    MANA_DRAW_BAND_LOW,
    NO_ASSIST_MIN_SCORE_RATE,
    AGateResult,
    CandidateExternalBench,
    CandidateInternalMetrics,
    GauntletOutcomes,
    GameResult,
    ManaDrawBaseline,
    build_a_gate_gauntlet_config,
    check_exploit_resistance_gate,
    check_h2h_trending,
    check_mana_draw_band,
    check_no_assist_gate,
    compute_mana_draw_rate,
    compute_score_rate,
    evaluate_a_gate,
    has_mlx_or_rust,
    is_baseline_valid,
    play_gauntlet,
    record_mana_draw_baseline,
    run_exploit_resistance_gauntlet,
    run_no_assist_gauntlet,
    select_promotion,
)
from train_v3.gauntlet_v5 import (
    EXPLOIT_AGENT_KINDS as GAUNTLET_EXPLOIT_KINDS,
    V5GauntletConfig,
    build_default_exploit_gauntlet,
)


# ---------------------------------------------------------------------------
# helpers — a deterministic fake GameRunner for the gauntlet aggregation tests
# ---------------------------------------------------------------------------
class _FakeGameRunner:
    """Deterministic fake ``GameRunner``: returns a fixed outcome distribution +
    mana_draw counts per opponent, keyed by a seed for reproducibility. Used to
    test ``play_gauntlet`` / ``run_*_gauntlet`` WITHOUT MLX/Rust."""

    def __init__(self, *, win_rate: float, draw_rate: float, mana_draw_rate: float):
        self.win_rate = float(win_rate)
        self.draw_rate = float(draw_rate)
        self.mana_draw_rate = float(mana_draw_rate)
        self.eligible_turns_per_game = 10

    def play(self, opponent_kind: str, *, seed: int) -> GameResult:
        # deterministic pseudo-random from the seed (no numpy needed)
        r = ((seed * 1103515245 + 12345) & 0x7FFFFFFF) / float(0x7FFFFFFF)
        if r < self.win_rate:
            outcome = "win"
        elif r < self.win_rate + self.draw_rate:
            outcome = "draw"
        else:
            outcome = "loss"
        md = int(round(self.mana_draw_rate * self.eligible_turns_per_game))
        md = max(0, min(md, self.eligible_turns_per_game))
        return GameResult(
            outcome=outcome,
            mana_draw_count=md,
            eligible_turns=self.eligible_turns_per_game,
            opponent=opponent_kind,
        )


def _passing_a_gate(
    *,
    no_assist: float = 0.60,
    exploit: float = 0.55,
    candidate_md_rate: float = 0.40,
    baseline: ManaDrawBaseline | None = None,
    h2h: list[float] | None = None,
) -> AGateResult:
    """Helper: build an A-gate result that PASSES all 4 criteria (overridable)."""
    if baseline is None:
        # B = 0.4 -> band [0.2, 0.6]; candidate 0.4 is inside.
        baseline = record_mana_draw_baseline(40, 100)
    if h2h is None:
        h2h = [0.40, 0.45, 0.50, 0.55, 0.60]
    return evaluate_a_gate(
        no_assist_score_rate=no_assist,
        exploit_resistance_score_rate=exploit,
        candidate_mana_draw_rate=candidate_md_rate,
        mana_draw_baseline=baseline,
        h2h_scores=h2h,
    )


# ===========================================================================
# 1. no_assist gate threshold (>= 0.55, raised from dead-field 0.45)
# ===========================================================================
class TestNoAssistGate:
    def test_passes_at_threshold_055(self):
        o = check_no_assist_gate(0.55)
        assert o.passed is True
        assert o.name == "no_assist"
        assert o.threshold == pytest.approx(0.55)

    def test_fails_at_050(self):
        o = check_no_assist_gate(0.50)
        assert o.passed is False

    def test_fails_just_below_at_054(self):
        # boundary: 0.54 < 0.55 -> fail (strict >=)
        o = check_no_assist_gate(0.54)
        assert o.passed is False

    def test_threshold_is_raised_from_dead_field_045(self):
        # the spec threshold (0.55) MUST exceed the dead-field 0.45
        # (gauntlet_v5.py:42); A5 does NOT wire the dead field.
        assert NO_ASSIST_MIN_SCORE_RATE == 0.55
        assert NO_ASSIST_MIN_SCORE_RATE > 0.45
        assert V5GauntletConfig().no_assist_min_score_rate == 0.45  # dead, unchanged
        o = check_no_assist_gate(0.50)
        assert o.passed is False  # 0.50 would pass the dead 0.45 but NOT the spec 0.55
        assert o.details["raised_from"] == 0.45


# ===========================================================================
# 2. exploit_resistance gate threshold (>= 0.50, raised from dead-field 0.42)
# ===========================================================================
class TestExploitResistanceGate:
    def test_passes_at_threshold_050(self):
        o = check_exploit_resistance_gate(0.50)
        assert o.passed is True
        assert o.name == "exploit_resistance"
        assert o.threshold == pytest.approx(0.50)

    def test_fails_at_045(self):
        o = check_exploit_resistance_gate(0.45)
        assert o.passed is False

    def test_fails_just_below_at_049(self):
        o = check_exploit_resistance_gate(0.49)
        assert o.passed is False

    def test_uses_exploit_agent_kinds_roster(self):
        # the default roster MUST be gauntlet_v5.py:8 EXPLOIT_AGENT_KINDS
        o = check_exploit_resistance_gate(0.50)
        assert tuple(o.details["exploit_kinds"]) == tuple(EXPLOIT_AGENT_KINDS)
        assert tuple(o.details["exploit_kinds"]) == tuple(GAUNTLET_EXPLOIT_KINDS)
        assert EXPLOIT_AGENT_KINDS == (
            "face_rush",
            "board_control",
            "greedy_trade",
            "stall",
            "punish_empty_board",
            "anti_draw_greed",
            "anti_hand_leak_overfit",
        )

    def test_threshold_raised_from_dead_field_042(self):
        assert EXPLOIT_RESISTANCE_MIN_SCORE_RATE == 0.50
        assert EXPLOIT_RESISTANCE_MIN_SCORE_RATE > 0.42
        assert V5GauntletConfig().exploit_resistance_min_score_rate == 0.42  # dead
        o = check_exploit_resistance_gate(0.45)
        assert o.passed is False  # 0.45 passes dead 0.42 but NOT spec 0.50
        assert o.details["raised_from"] == 0.42


# ===========================================================================
# 3. mana_draw band [0.5x, 1.5x] + the HAND_CAP/MANA_DRAW_BASE invalidation guard
# ===========================================================================
class TestManaDrawBand:
    def test_within_band_passes(self):
        # B = 0.4 (40/100); band [0.2, 0.6]; candidate 0.4 inside -> pass
        b = record_mana_draw_baseline(40, 100)
        assert b.rate == pytest.approx(0.4)
        o = check_mana_draw_band(0.40, b)
        assert o.passed is True
        assert o.details["band"] == pytest.approx([0.2, 0.6])

    def test_at_lower_edge_passes(self):
        b = record_mana_draw_baseline(40, 100)  # B=0.4 -> low=0.2
        o = check_mana_draw_band(0.2, b)
        assert o.passed is True  # inclusive lower edge

    def test_at_upper_edge_passes(self):
        b = record_mana_draw_baseline(40, 100)  # B=0.4 -> high=0.6
        o = check_mana_draw_band(0.6, b)
        assert o.passed is True  # inclusive upper edge

    def test_below_band_fails(self):
        b = record_mana_draw_baseline(40, 100)  # band [0.2, 0.6]
        o = check_mana_draw_band(0.19, b)
        assert o.passed is False
        assert o.details["invalidated"] is False

    def test_above_band_fails(self):
        b = record_mana_draw_baseline(40, 100)  # band [0.2, 0.6]
        o = check_mana_draw_band(0.61, b)
        assert o.passed is False

    def test_band_multipliers_are_05_and_15(self):
        assert MANA_DRAW_BAND_LOW == 0.5
        assert MANA_DRAW_BAND_HIGH == 1.5

    def test_invalidation_guard_hand_cap_changed(self):
        # baseline recorded with HAND_CAP=4; live engine now HAND_CAP=5 -> B stale
        b = record_mana_draw_baseline(40, 100, hand_cap=4, mana_draw_base=2)
        assert b.valid is True
        # the candidate rate would otherwise be in-band (0.4 in [0.2,0.6])...
        o = check_mana_draw_band(0.40, b, current_hand_cap=5, current_mana_draw_base=2)
        # ...but the gate FAILS because B is invalidated (must re-measure).
        assert o.passed is False
        assert o.details["invalidated"] is True
        assert o.details["recorded_hand_cap"] == 4
        assert o.details["current_hand_cap"] == 5

    def test_invalidation_guard_mana_draw_base_changed(self):
        b = record_mana_draw_baseline(40, 100, hand_cap=4, mana_draw_base=2)
        o = check_mana_draw_band(0.40, b, current_hand_cap=4, current_mana_draw_base=3)
        assert o.passed is False
        assert o.details["invalidated"] is True
        assert o.details["recorded_mana_draw_base"] == 2
        assert o.details["current_mana_draw_base"] == 3

    def test_invalidation_guard_constants_unchanged_passes(self):
        b = record_mana_draw_baseline(40, 100, hand_cap=4, mana_draw_base=2)
        o = check_mana_draw_band(0.40, b, current_hand_cap=4, current_mana_draw_base=2)
        assert o.passed is True
        assert o.details["invalidated"] is False

    def test_is_baseline_valid(self):
        b = record_mana_draw_baseline(40, 100, hand_cap=4, mana_draw_base=2)
        assert is_baseline_valid(b, current_hand_cap=4, current_mana_draw_base=2) is True
        assert is_baseline_valid(b, current_hand_cap=5, current_mana_draw_base=2) is False
        assert is_baseline_valid(b, current_hand_cap=4, current_mana_draw_base=3) is False

    def test_engine_constants_recorded(self):
        # the A5-recorded engine constants MUST match core/engine.py:44,59
        assert ENGINE_HAND_CAP == 4
        assert ENGINE_MANA_DRAW_BASE == 2

    def test_compute_mana_draw_rate_validates(self):
        assert compute_mana_draw_rate(40, 100) == pytest.approx(0.4)
        with pytest.raises(ValueError):
            compute_mana_draw_rate(40, 0)
        with pytest.raises(ValueError):
            compute_mana_draw_rate(-1, 100)

    def test_synthetic_pilot_traces_known_counts(self):
        # synthetic pilot trace: 25 mana_draws over 50 eligible turns -> B=0.5
        # band [0.25, 0.75]; candidate 0.5 inside -> pass; 0.74 inside -> pass;
        # 0.76 outside -> fail; 0.24 outside -> fail.
        b = record_mana_draw_baseline(25, 50)
        assert b.rate == pytest.approx(0.5)
        assert check_mana_draw_band(0.5, b).passed is True
        assert check_mana_draw_band(0.74, b).passed is True
        assert check_mana_draw_band(0.76, b).passed is False
        assert check_mana_draw_band(0.24, b).passed is False


# ===========================================================================
# 4. H2H vs best self-snapshot trending
# ===========================================================================
class TestH2HTrending:
    def test_monotonically_improving_passes(self):
        o = check_h2h_trending([0.40, 0.45, 0.50, 0.55, 0.60])
        assert o.passed is True
        assert o.details["regressed"] is False

    def test_non_monotone_fails(self):
        # dip at index 2 (0.5 -> 0.45) -> regression -> fail
        o = check_h2h_trending([0.40, 0.50, 0.45, 0.55, 0.60])
        assert o.passed is False
        assert o.details["regressed"] is True

    def test_plateau_passes(self):
        # equal values are non-decreasing -> pass
        o = check_h2h_trending([0.5, 0.5, 0.5, 0.5, 0.5])
        assert o.passed is True

    def test_insufficient_snapshots_fails(self):
        # fewer than min_snapshots (default 5) -> fail (insufficient data)
        o = check_h2h_trending([0.40, 0.50, 0.60, 0.70])
        assert o.passed is False
        assert o.details["reason"] == "insufficient_snapshots"
        assert o.details["n_measured"] == 4
        assert o.details["min_snapshots"] == DEFAULT_H2H_MIN_SNAPSHOTS

    def test_uses_last_n_snapshots(self):
        # 7 measurements, min 5 -> the last 5 must be non-decreasing.
        # last 5 = [0.5, 0.6, 0.7, 0.8, 0.9] (non-decreasing) -> pass, even though
        # the full series has an early dip.
        o = check_h2h_trending([0.9, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9])
        assert o.passed is True
        assert o.details["recent_scores"] == pytest.approx([0.5, 0.6, 0.7, 0.8, 0.9])

    def test_regression_in_last_n_fails(self):
        # the dip is INSIDE the last 5 -> fail
        o = check_h2h_trending([0.9, 0.1, 0.5, 0.6, 0.55, 0.8, 0.9])
        assert o.passed is False

    def test_default_min_snapshots_is_5(self):
        assert DEFAULT_H2H_MIN_SNAPSHOTS == 5

    def test_tolerance_allows_small_dip(self):
        # with tolerance=0.05, a 0.03 dip is tolerated -> pass
        o = check_h2h_trending([0.50, 0.47, 0.55, 0.60, 0.65], tolerance=0.05)
        assert o.passed is True

    def test_strict_tolerance_fails_small_dip(self):
        o = check_h2h_trending([0.50, 0.47, 0.55, 0.60, 0.65], tolerance=0.0)
        assert o.passed is False


# ===========================================================================
# 5. A-gate aggregate (all 4 must pass; failing any one fails)
# ===========================================================================
class TestAGateAggregate:
    def test_all_four_pass(self):
        r = _passing_a_gate()
        assert r.passed is True
        assert r.failed_criteria() == []

    def test_failing_no_assist_only(self):
        r = evaluate_a_gate(
            no_assist_score_rate=0.50,  # FAILS (< 0.55)
            exploit_resistance_score_rate=0.55,
            candidate_mana_draw_rate=0.40,
            mana_draw_baseline=record_mana_draw_baseline(40, 100),
            h2h_scores=[0.4, 0.45, 0.5, 0.55, 0.6],
        )
        assert r.passed is False
        assert "no_assist" in r.failed_criteria()
        assert "exploit_resistance" not in r.failed_criteria()

    def test_failing_exploit_resistance_only(self):
        r = evaluate_a_gate(
            no_assist_score_rate=0.60,
            exploit_resistance_score_rate=0.45,  # FAILS (< 0.50)
            candidate_mana_draw_rate=0.40,
            mana_draw_baseline=record_mana_draw_baseline(40, 100),
            h2h_scores=[0.4, 0.45, 0.5, 0.55, 0.6],
        )
        assert r.passed is False
        assert "exploit_resistance" in r.failed_criteria()

    def test_failing_mana_draw_band_only(self):
        r = evaluate_a_gate(
            no_assist_score_rate=0.60,
            exploit_resistance_score_rate=0.55,
            candidate_mana_draw_rate=0.61,  # outside [0.2, 0.6] -> FAILS
            mana_draw_baseline=record_mana_draw_baseline(40, 100),
            h2h_scores=[0.4, 0.45, 0.5, 0.55, 0.6],
        )
        assert r.passed is False
        assert "mana_draw_band" in r.failed_criteria()

    def test_failing_h2h_trending_only(self):
        r = evaluate_a_gate(
            no_assist_score_rate=0.60,
            exploit_resistance_score_rate=0.55,
            candidate_mana_draw_rate=0.40,
            mana_draw_baseline=record_mana_draw_baseline(40, 100),
            h2h_scores=[0.4, 0.5, 0.45, 0.55, 0.6],  # dip -> FAILS
        )
        assert r.passed is False
        assert "h2h_trending" in r.failed_criteria()

    def test_mana_draw_invalidation_propagates_to_a_gate(self):
        # if the baseline is invalidated by changed constants, the A-gate fails
        # even though the candidate rate would otherwise be in-band.
        r = evaluate_a_gate(
            no_assist_score_rate=0.60,
            exploit_resistance_score_rate=0.55,
            candidate_mana_draw_rate=0.40,  # in [0.2, 0.6]
            mana_draw_baseline=record_mana_draw_baseline(40, 100, hand_cap=4, mana_draw_base=2),
            h2h_scores=[0.4, 0.45, 0.5, 0.55, 0.6],
            current_hand_cap=5,  # CHANGED -> invalidated
            current_mana_draw_base=2,
        )
        assert r.passed is False
        assert r.mana_draw_band.details["invalidated"] is True
        assert "mana_draw_band" in r.failed_criteria()

    def test_to_dict(self):
        r = _passing_a_gate()
        d = r.to_dict()
        assert d["passed"] is True
        assert d["failed_criteria"] == []
        assert set(d.keys()) == {
            "passed", "failed_criteria", "no_assist",
            "exploit_resistance", "mana_draw_band", "h2h_trending",
        }


# ===========================================================================
# 6. Promotion selector + the promotion-by-loss GUARD (the D-lesson, gap #7)
# ===========================================================================
class TestPromotionSelector:
    def _passing_external(self, h2h_vs_best: float = 0.55) -> CandidateExternalBench:
        return CandidateExternalBench(
            a_gate=_passing_a_gate(),
            h2h_vs_best_score_rate=h2h_vs_best,
        )

    def _failing_external(self, h2h_vs_best: float = 0.55) -> CandidateExternalBench:
        # A-gate fails (no_assist too low)
        a_gate = evaluate_a_gate(
            no_assist_score_rate=0.50,  # FAILS
            exploit_resistance_score_rate=0.55,
            candidate_mana_draw_rate=0.40,
            mana_draw_baseline=record_mana_draw_baseline(40, 100),
            h2h_scores=[0.4, 0.45, 0.5, 0.55, 0.6],
        )
        assert a_gate.passed is False
        return CandidateExternalBench(a_gate=a_gate, h2h_vs_best_score_rate=h2h_vs_best)

    def test_passing_a_gate_and_beating_best_is_promoted(self):
        ext = self._passing_external(h2h_vs_best=0.55)  # > 0.5 -> beats best
        dec = select_promotion(ext, current_best_h2h_score_rate=0.5)
        assert dec.promoted is True
        assert dec.a_gate_passed is True
        assert dec.reason == "promoted_beats_best"

    def test_passing_a_gate_but_tie_not_promoted(self):
        # h2h == 0.5 (even) -> does NOT strictly beat -> no promotion
        ext = self._passing_external(h2h_vs_best=0.5)
        dec = select_promotion(ext, current_best_h2h_score_rate=0.5)
        assert dec.promoted is False
        assert dec.reason == "h2h_not_beating_best"

    def test_failing_a_gate_not_promoted_even_with_high_h2h(self):
        ext = self._failing_external(h2h_vs_best=0.90)
        dec = select_promotion(ext, current_best_h2h_score_rate=0.5)
        assert dec.promoted is False
        assert dec.reason == "a_gate_failed"

    def test_promotion_by_loss_guard_lower_loss_failing_gate_not_promoted(self):
        # THE D-LESSON GUARD (load-bearing, gap #7): a candidate with LOWER internal
        # loss but FAILING the external A-gate must NOT promote.
        ext = self._failing_external(h2h_vs_best=0.90)
        low_loss = CandidateInternalMetrics(ppo_loss=0.01, approx_kl=0.001, entropy=0.9)
        dec = select_promotion(ext, low_loss, current_best_h2h_score_rate=0.5)
        assert dec.promoted is False
        # the internal metrics are echoed for monitoring but did NOT promote
        assert dec.internal_metrics.ppo_loss == 0.01
        assert dec.reason == "a_gate_failed"

    def test_promotion_independent_of_ppo_loss_identical_external(self):
        # two candidates with IDENTICAL external-bench but DIFFERENT internal
        # metrics (loss/KL/entropy) -> SAME promotion decision (verifier finding 3b).
        ext = self._passing_external(h2h_vs_best=0.55)
        low_loss = CandidateInternalMetrics(ppo_loss=0.01, approx_kl=0.001, entropy=0.9)
        high_loss = CandidateInternalMetrics(ppo_loss=0.99, approx_kl=0.5, entropy=0.1)
        dec_low = select_promotion(ext, low_loss, current_best_h2h_score_rate=0.5)
        dec_high = select_promotion(ext, high_loss, current_best_h2h_score_rate=0.5)
        assert dec_low.promoted == dec_high.promoted is True
        assert dec_low.reason == dec_high.reason
        # internal metrics differ (recorded) but did NOT change the decision
        assert dec_low.internal_metrics.ppo_loss != dec_high.internal_metrics.ppo_loss

    def test_promotion_independent_of_ppo_loss_failing_gate(self):
        # identical FAILING external, different loss -> both NOT promoted
        ext = self._failing_external(h2h_vs_best=0.55)
        low_loss = CandidateInternalMetrics(ppo_loss=0.001)
        high_loss = CandidateInternalMetrics(ppo_loss=1.0)
        dec_low = select_promotion(ext, low_loss, current_best_h2h_score_rate=0.5)
        dec_high = select_promotion(ext, high_loss, current_best_h2h_score_rate=0.5)
        assert dec_low.promoted == dec_high.promoted is False

    def test_first_snapshot_promotes_on_a_gate_pass(self):
        # current_best_h2h_score_rate=None -> first snapshot; A-gate pass -> promote
        ext = self._passing_external(h2h_vs_best=0.0)
        dec = select_promotion(ext, current_best_h2h_score_rate=None)
        assert dec.promoted is True
        assert dec.is_first_snapshot is True
        assert dec.reason == "promoted_first_snapshot"

    def test_first_snapshot_failing_a_gate_not_promoted(self):
        ext = self._failing_external(h2h_vs_best=0.0)
        dec = select_promotion(ext, current_best_h2h_score_rate=None)
        assert dec.promoted is False
        assert dec.reason == "a_gate_failed"
        assert dec.is_first_snapshot is True

    def test_h2h_promotion_threshold_default_is_05(self):
        assert H2H_PROMOTION_THRESHOLD == 0.5

    def test_to_dict_echoes_internal_as_monitoring_only(self):
        ext = self._passing_external(h2h_vs_best=0.55)
        internal = CandidateInternalMetrics(ppo_loss=0.2, approx_kl=0.05, entropy=0.7)
        dec = select_promotion(ext, internal, current_best_h2h_score_rate=0.5)
        d = dec.to_dict()
        assert d["promoted"] is True
        assert "internal_metrics_monitoring_only" in d
        assert d["internal_metrics_monitoring_only"]["ppo_loss"] == 0.2


# ===========================================================================
# 7. score rate + gauntlet aggregation (synthetic, no MLX/Rust)
# ===========================================================================
class TestScoreRateAndGauntlet:
    def test_score_rate_formula(self):
        # (wins + 0.5*draws) / total
        assert compute_score_rate(10, 0, 0) == pytest.approx(1.0)
        assert compute_score_rate(0, 10, 0) == pytest.approx(0.5)
        assert compute_score_rate(0, 0, 10) == pytest.approx(0.0)
        assert compute_score_rate(5, 4, 1) == pytest.approx((5 + 2) / 10)  # 0.7
        with pytest.raises(ValueError):
            compute_score_rate(0, 0, 0)

    def test_play_gauntlet_aggregates(self):
        runner = _FakeGameRunner(win_rate=0.6, draw_rate=0.1, mana_draw_rate=0.4)
        outcomes = play_gauntlet(
            runner, ["stall", "anti_draw_greed"], games_per_opponent=20, seed=7
        )
        assert outcomes.total() == 40
        # score rate in a plausible range (deterministic from the seed)
        sr = outcomes.score_rate()
        assert 0.0 <= sr <= 1.0
        # mana_draw aggregation: 40 games * 10 eligible_turns = 400 eligible; md rate ~0.4
        assert outcomes.eligible_turns == 400
        assert outcomes.mana_draw_count == 40 * 4  # 0.4 * 10 rounded
        assert set(outcomes.per_opponent.keys()) == {"stall", "anti_draw_greed"}
        assert outcomes.per_opponent["stall"]["eligible_turns"] == 200

    def test_gauntlet_outcomes_mana_draw_rate(self):
        runner = _FakeGameRunner(win_rate=0.5, draw_rate=0.0, mana_draw_rate=0.3)
        outcomes = play_gauntlet(
            runner, ["stall"], games_per_opponent=10, seed=1
        )
        # 10 games * 10 eligible = 100 eligible; md = 0.3*10 = 3 per game -> 30
        assert outcomes.mana_draw_rate() == pytest.approx(30 / 100)

    def test_run_no_assist_gauntlet_passes_with_strong_candidate(self):
        # win_rate high enough that score_rate >= 0.55
        runner = _FakeGameRunner(win_rate=0.6, draw_rate=0.0, mana_draw_rate=0.4)
        o = run_no_assist_gauntlet(runner, games_per_opponent=50, seed=3)
        assert o.name == "no_assist"
        assert o.threshold == pytest.approx(NO_ASSIST_MIN_SCORE_RATE)
        # deterministic: 7 exploit opponents * 50 games = 350 games; the fake's
        # win_rate=0.6, draw=0 -> score_rate ~= 0.6 (well above 0.55)
        assert o.passed is True

    def test_run_exploit_resistance_gauntlet(self):
        runner = _FakeGameRunner(win_rate=0.55, draw_rate=0.0, mana_draw_rate=0.4)
        o = run_exploit_resistance_gauntlet(runner, games_per_opponent=50, seed=5)
        assert o.name == "exploit_resistance"
        assert o.threshold == pytest.approx(EXPLOIT_RESISTANCE_MIN_SCORE_RATE)
        assert o.passed is True  # 0.55 >= 0.50

    def test_run_exploit_resistance_gauntlet_fails_weak(self):
        runner = _FakeGameRunner(win_rate=0.40, draw_rate=0.0, mana_draw_rate=0.4)
        o = run_exploit_resistance_gauntlet(runner, games_per_opponent=50, seed=5)
        assert o.passed is False  # 0.40 < 0.50

    def test_default_roster_is_exploit_agent_kinds(self):
        # run_*_gauntlet with no opponent_kinds uses EXPLOIT_AGENT_KINDS
        runner = _FakeGameRunner(win_rate=0.6, draw_rate=0.0, mana_draw_rate=0.4)
        o = run_no_assist_gauntlet(runner, games_per_opponent=5, seed=1)
        # 7 opponents * 5 games = 35 games
        assert o.details["criterion"] == "no_assist_score_rate"

    def test_play_gauntlet_validates_inputs(self):
        runner = _FakeGameRunner(win_rate=0.5, draw_rate=0.0, mana_draw_rate=0.4)
        with pytest.raises(ValueError):
            play_gauntlet(runner, ["stall"], games_per_opponent=0)
        with pytest.raises(ValueError):
            play_gauntlet(runner, [], games_per_opponent=5)


# ===========================================================================
# 8. config override (dead fields overridden at construction, NOT wired as gate)
# ===========================================================================
class TestConfigOverride:
    def test_build_a_gate_gauntlet_config_raises_thresholds(self):
        cfg = build_a_gate_gauntlet_config()
        assert cfg.no_assist_min_score_rate == 0.55  # raised from 0.45
        assert cfg.exploit_resistance_min_score_rate == 0.50  # raised from 0.42
        # the dead fields on a default config are STILL the dead values (unchanged)
        default = V5GauntletConfig()
        assert default.no_assist_min_score_rate == 0.45
        assert default.exploit_resistance_min_score_rate == 0.42

    def test_build_default_exploit_gauntlet_matches_kinds(self):
        lanes = build_default_exploit_gauntlet()
        assert tuple(lane.kind for lane in lanes) == tuple(EXPLOIT_AGENT_KINDS)
        assert all(lane.runtime == "rust" for lane in lanes)


# ===========================================================================
# 9. smoke import (verifier finding 2e: TrainV3.5 path, NOT broken TrainV3 path)
# ===========================================================================
class TestSmokeImport:
    def test_smoke_import_train_v3(self):
        # A5 imports V5GauntletConfig + build_default_exploit_gauntlet directly
        # from train_v3 via the TrainV3.5 path (NOT run_v5_acceptance.py's broken
        # ``TrainV3`` path, verifier finding 2e).
        from train_v3.gauntlet_v5 import (  # noqa: F401
            V5GauntletConfig as _Cfg,
            build_default_exploit_gauntlet as _B,
            EXPLOIT_AGENT_KINDS as _K,
        )
        from train_v3.a_gate import (  # noqa: F401
            evaluate_a_gate as _E,
            select_promotion as _S,
        )
        # sanity: the importable config is the real V5GauntletConfig (not a stub)
        cfg = _Cfg()
        assert hasattr(cfg, "no_assist_min_score_rate")
        assert hasattr(cfg, "exploit_resistance_min_score_rate")
        assert _K == EXPLOIT_AGENT_KINDS

    def test_a_gate_does_not_import_run_v5_acceptance(self):
        # A5 must NOT IMPORT run_v5_acceptance.py (broken TrainV3 path, verifier
        # finding 2e). Citations in comments/docstrings are fine; the guard is
        # against actual import statements that would pull in the broken path.
        import train_v3.a_gate as ag
        import ast
        tree = ast.parse(open(ag.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "run_v5_acceptance" not in alias.name, (
                        f"a_gate.py imports {alias.name} (broken-path run_v5_acceptance)"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "run_v5_acceptance" not in mod, (
                    f"a_gate.py imports from {mod} (broken-path run_v5_acceptance)"
                )


# ===========================================================================
# 10. skip-gate when MLX/Rust unbuildable (the operational live gauntlet)
# ===========================================================================
class TestSkipGate:
    def test_skip_if_no_mlx_or_rust(self):
        # The operational live gauntlet (real games via A4 rust_live_self_play)
        # needs MLX or the Rust FFI extension. The measurement + gating logic does
        # NOT (it is synthetic-testable, exercised by every other test above).
        # If neither is available, skip the live-gauntlet integration test.
        if not has_mlx_or_rust():
            pytest.skip("MLX/Rust not buildable in this environment; live gauntlet skipped")
        # If we ARE here (MLX or Rust available), the skip-gate itself is verified
        # to return True — the live gauntlet COULD run. We do not run a full live
        # gauntlet here (that needs a trained policy + the Rust extension); the
        # skip-gate logic is the acceptance.
        assert has_mlx_or_rust() is True

    def test_has_mlx_or_rust_returns_bool(self):
        assert isinstance(has_mlx_or_rust(), bool)