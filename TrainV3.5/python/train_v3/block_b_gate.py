"""Block B component B6 -- ``block_b_gate.py`` -- Block-B external-bench promotion
gate (EXTENDS A5, does NOT rewrite ``a_gate.py``).

Purpose (``BLOCK_B_PLAN.md:502-522`` / ``design.md:121``): promote a candidate to
the new best-self-snapshot iff the FULL external-bench aggregate (H2H vs best
self-snapshot + gauntlet + mana_draw band + p1_p2 gap) improves MONOTONICALLY over
``N_snap`` consecutive snapshots (D-B1 default 5). Distinct from the A-gate
(``design.md:114``, the PHASE-A EXIT gate): Block-B promotion does NOT re-apply
``no_assist`` / ``exploit_resistance`` -- those are Phase-A exit criteria, NOT
league-promotion criteria (open_question #11/#12). The promotion-by-loss guard is
INHERITED from A5 ``select_promotion`` (``a_gate.py:607``, ``:638-641``): the
decision NEVER consults internal ``ppo_loss`` / ``approx_kl`` / ``entropy``
(``CandidateInternalMetrics`` is monitoring-only); the deliberate absence of any
read of those fields IS the guard.

Composition (A5 = oracle, B6 = UUT -- composes A5 pieces, builds NEW
``BlockBGateResult`` + monotone aggregate; does NOT mutate A5 ``AGateResult``):
  * REUSES A5 ``GateOutcome`` (``a_gate.py:151``) for each Block-B component.
  * REUSES A5 ``check_mana_draw_band`` (``a_gate.py:309``) for the mana_draw-band
    component (calls it with the candidate's measured mana_draw rate + baseline).
  * REUSES A5 ``play_gauntlet`` (``a_gate.py:753``) via ``measure_gauntlet_rate``
    for the gauntlet component (the promotion-bench gauntlet; B5's side-stratified
    run is separate -- B6 consumes B5's ``gap_for_promotion`` for the parity term).
  * Does NOT call ``check_no_assist_gate`` / ``check_exploit_resistance_gate`` /
    ``evaluate_a_gate`` (regression guard: Block B does NOT re-apply the Phase-A
    exit criteria).

A5 ``check_h2h_trending`` (``a_gate.py:374``) checks H2H non-decreasing (trending
up). B6 needs MONOTONE IMPROVEMENT of the FULL 4-component aggregate -- a
DISTINCT check (all 4 components, not just H2H). B6 builds its own
monotone-aggregate check; the promotion verdict uses the full aggregate
monotonicity, NOT ``check_h2h_trending`` alone.

B5/B1 grounding: B5 ``SecondStartParityLoop.gap_for_promotion`` (``second_start_
parity.py:239``) PRODUCES the measured p1/p2 gap; B6 CONSUMES it (B6 does NOT
depend on B7). B1 ``SnapshotPool`` best-ever anchor (``snapshot_pool.py:87,
BEST_EVER_ROLE``) is the H2H-vs-best reference; for synthetic tests B6 takes the
H2H rate as a measured input (the caller harvests it vs the best-ever snapshot).

frozen-classic guard: no edit to ``classic_*`` / ``reward_v5`` / ``v5_trace`` /
``warm_start_v5`` / ``run_phase26*`` / ``run_v5_acceptance`` / ``league_v5.py`` /
``gauntlet_v5.py`` / ``opponents_v5.py`` (read-only). NO edit to A5 ``a_gate.py``,
B5 ``second_start_parity.py``, B1 ``snapshot_pool.py`` (compose/consume read-only).
NO Rust edit. NO TrainV3.5-into-prod. Synthetic tests only (fake runner +
fabricated series, no real Rust/MLX/ONNX).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A5 = oracle. REUSE (read-only) the A5 pieces B6 composes:
# ``GateOutcome`` (the per-criterion result type), ``check_mana_draw_band`` (the
# mana_draw-band criterion, ``a_gate.py:309``), ``play_gauntlet`` (the gauntlet
# runner, ``a_gate.py:753``), ``ManaDrawBaseline`` (Q4 baseline B), ``GameRunner``
# (the injectable runner Protocol), ``CandidateInternalMetrics`` (monitoring-only
# internal metrics, carried but NEVER read -- the inherited promotion-by-loss
# guard), and the engine constants + band defaults.
from .a_gate import (
    MANA_DRAW_BAND_HIGH,
    MANA_DRAW_BAND_LOW,
    CandidateInternalMetrics,
    GameRunner,
    GateOutcome,
    ManaDrawBaseline,
    play_gauntlet,
    check_mana_draw_band,
)

#: D-B1 default -- number of consecutive snapshots over which the full
#: external-bench aggregate must improve monotonically to promote
#: (``BLOCK_B_PLAN.md:159``, ``design.md:121``).
DEFAULT_BLOCK_B_N_SNAP: int = 5

#: p1/p2 score-gap threshold (continuous parity, ``design.md:121``). B5
#: ``SecondStartParityLoop.breach`` uses ``gap > 0.12`` (``second_start_parity.py:
#: 236`` ``PHASE_A_P1_P2_GAP_THRESHOLD``); B6 consumes the same value as the
#: promotion criterion (gap <= 0.12 required to promote).
P1_P2_GAP_THRESHOLD: float = 0.12

#: H2H-vs-best score-rate threshold: the candidate's H2H score rate vs the
#: current best-ever self-snapshot must be >= this for the H2H component to pass
#: (mirrors A5 ``H2H_PROMOTION_THRESHOLD`` = 0.5 = even; a tie = no improvement).
BLOCK_B_H2H_MIN_SCORE_RATE: float = 0.5

#: Gauntlet score-rate threshold: the candidate's gauntlet score rate (A5
#: ``play_gauntlet`` aggregate, ``a_gate.py:753``) must be >= this for the
#: gauntlet component to pass. Default 0.5 (the promotion-bench gauntlet is a
#: generic skill probe, not the spec-raised no_assist/exploit_resistance gates
#: which are Phase-A exit criteria Block B does NOT re-apply).
BLOCK_B_GAUNTLET_MIN_SCORE_RATE: float = 0.5

#: Per-step tolerance for the monotone-aggregate check. Default 0.0 = strict
#: non-decreasing (synthetic tests use 0.0; a small tolerance allows noisy real
#: measurements to count as monotone).
DEFAULT_MONOTONE_TOLERANCE: float = 0.0


def block_b_aggregate(
    h2h_rate: float,
    gauntlet_rate: float,
    mana_draw_in_band: bool,
    p1_p2_gap: float,
) -> float:
    """The composite external-bench scalar (higher = better).

    Formula (``design.md:121`` -- the 4-component external-bench aggregate):

        aggregate = h2h_rate
                  + gauntlet_rate
                  + (1.0 if mana_draw_in_band else 0.0)
                  + max(0.0, 1.0 - p1_p2_gap / P1_P2_GAP_THRESHOLD)

    Each rate term is in ``[0, 1]``; the mana_draw-band term is a 0/1 indicator
    (1.0 when the candidate's mana_draw rate is in the ``[0.5x, 1.5x]`` band of
    the baseline B, 0.0 otherwise); the parity term is ``1.0 - gap/0.12`` clamped
    to ``[0, 1]`` -- it is ``1.0`` at gap=0, ``0.0`` at gap=0.12, and ``0.0`` for
    any gap > 0.12 (a parity breach dips the aggregate, so a gap breach cannot
    monotonically promote). Range: ``[0.0, 3.0]``.

    ``p1_p2_gap`` values above ``P1_P2_GAP_THRESHOLD`` (a) fail the ``p1_p2_gap``
    component gate (so ``all_four_pass`` is False -> no promote regardless of
    monotonicity) AND (b) lower the parity term to 0 (the clamp), which drops the
    aggregate and makes the monotone window harder to keep non-decreasing. A gap
    breach therefore cannot promote (the component gate alone blocks it; the
    parity-term dip compounds the effect).
    """
    h2h_rate = float(h2h_rate)
    gauntlet_rate = float(gauntlet_rate)
    parity_term = max(0.0, 1.0 - float(p1_p2_gap) / float(P1_P2_GAP_THRESHOLD))
    return h2h_rate + gauntlet_rate + (1.0 if mana_draw_in_band else 0.0) + parity_term


@dataclass(frozen=True)
class BlockBGateResult:
    """The full Block-B promotion verdict (NEW frozen dataclass; does NOT wrap
    A5 ``AGateResult`` -- ``AGateResult`` carries ``no_assist`` /
    ``exploit_resistance`` which Block B does NOT re-apply).

    The 4 Block-B external-bench components (each a reused A5 ``GateOutcome``):
      * ``h2h_vs_best`` -- H2H score rate vs the best-ever self-snapshot.
      * ``gauntlet`` -- gauntlet score rate (A5 ``play_gauntlet`` aggregate).
      * ``mana_draw_band`` -- mana_draw-usage band (A5 ``check_mana_draw_band``).
      * ``p1_p2_gap`` -- p1/p2 score-gap parity (B5 ``gap_for_promotion``).

    ``passed`` is the monotone-promotion verdict: True iff
    ``len(monotone_aggregate_history) >= n_snap`` AND the last ``n_snap``
    aggregates are non-decreasing within ``tolerance`` AND all 4 components pass
    for the most-recent snapshot. NO ``no_assist`` / ``exploit_resistance`` fields
    (regression guard: Block B does NOT re-apply the Phase-A exit criteria).

    ``reason`` is the verdict discriminator (one of ``'promoted'``,
    ``'insufficient_snapshots'``, ``'monotone_not_improving'``,
    ``'component_failed'``): a caller can distinguish a not-yet-promote due to too
    few snapshots (``'insufficient_snapshots'``) from a monotone failure
    (``'monotone_not_improving'``) from a failing component
    (``'component_failed'``) without re-deriving from
    ``len(monotone_aggregate_history)`` + ``failed_criteria()``.
    """

    h2h_vs_best: GateOutcome
    gauntlet: GateOutcome
    mana_draw_band: GateOutcome
    p1_p2_gap: GateOutcome
    passed: bool
    monotone_aggregate_history: tuple[float, ...]
    n_snap: int
    reason: str

    def failed_criteria(self) -> list[str]:
        """Names of the 4 Block-B components that failed for the most-recent
        snapshot (empty iff all 4 pass -- necessary but not sufficient for
        ``passed``; ``passed`` also requires the monotone-aggregate window)."""
        return [
            g.name
            for g in (
                self.h2h_vs_best,
                self.gauntlet,
                self.mana_draw_band,
                self.p1_p2_gap,
            )
            if not g.passed
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "reason": str(self.reason),
            "failed_criteria": self.failed_criteria(),
            "n_snap": int(self.n_snap),
            "monotone_aggregate_history": list(self.monotone_aggregate_history),
            "h2h_vs_best": _outcome_dict(self.h2h_vs_best),
            "gauntlet": _outcome_dict(self.gauntlet),
            "mana_draw_band": _outcome_dict(self.mana_draw_band),
            "p1_p2_gap": _outcome_dict(self.p1_p2_gap),
        }


def _outcome_dict(o: GateOutcome) -> dict[str, Any]:
    return {
        "name": o.name,
        "passed": bool(o.passed),
        "score": float(o.score),
        "threshold": float(o.threshold),
        "details": dict(o.details),
    }


def _check_h2h_vs_best(
    h2h_rate: float,
    *,
    threshold: float = BLOCK_B_H2H_MIN_SCORE_RATE,
) -> GateOutcome:
    """H2H-vs-best component: records the candidate's H2H score rate vs the
    best-ever self-snapshot; passes iff ``h2h_rate >= threshold`` (default 0.5 =
    even). The monotone-improvement verdict is the full-aggregate check, NOT this
    component alone -- this component only gates the most-recent snapshot."""
    h2h_rate = float(h2h_rate)
    passed = h2h_rate >= float(threshold)
    return GateOutcome(
        name="h2h_vs_best",
        passed=passed,
        score=h2h_rate,
        threshold=float(threshold),
        details={
            "criterion": "h2h_vs_best_self_snapshot",
            "best_ever_ref": "snapshot_pool.BEST_EVER_ROLE",
            "spec_source": "design.md:121",
        },
    )


def _check_gauntlet(
    gauntlet_rate: float,
    *,
    threshold: float = BLOCK_B_GAUNTLET_MIN_SCORE_RATE,
) -> GateOutcome:
    """Gauntlet component: records the candidate's gauntlet score rate (A5
    ``play_gauntlet`` aggregate); passes iff ``gauntlet_rate >= threshold``
    (default 0.5). This is the promotion-bench gauntlet, NOT the Phase-A
    no_assist / exploit_resistance gauntlets (Block B does NOT re-apply those)."""
    gauntlet_rate = float(gauntlet_rate)
    passed = gauntlet_rate >= float(threshold)
    return GateOutcome(
        name="gauntlet",
        passed=passed,
        score=gauntlet_rate,
        threshold=float(threshold),
        details={
            "criterion": "gauntlet_score_rate",
            "runner": "a_gate.play_gauntlet",
            "spec_source": "design.md:121",
        },
    )


def _check_p1_p2_gap(
    p1_p2_gap: float,
    *,
    threshold: float = P1_P2_GAP_THRESHOLD,
) -> GateOutcome:
    """p1/p2 gap component (B5 ``gap_for_promotion``): passes iff
    ``p1_p2_gap <= threshold`` (default 0.12, continuous parity,
    ``design.md:121``). A gap breach (> 0.12) fails this component AND dips the
    aggregate parity term to 0 (so a breach cannot monotonically promote)."""
    p1_p2_gap = float(p1_p2_gap)
    passed = p1_p2_gap <= float(threshold)
    return GateOutcome(
        name="p1_p2_gap",
        passed=passed,
        score=p1_p2_gap,
        threshold=float(threshold),
        details={
            "criterion": "p1_p2_score_gap_parity",
            "source": "second_start_parity.SecondStartParityLoop.gap_for_promotion",
            "spec_source": "design.md:121",
        },
    )


def _monotone_non_decreasing(
    series: list[float] | tuple[float, ...],
    *,
    tolerance: float,
) -> bool:
    """True iff ``series`` is non-decreasing within ``tolerance`` (each step >=
    previous - tolerance). Empty / single-element series are trivially
    non-decreasing."""
    s = [float(x) for x in series]
    tol = float(tolerance)
    return all(s[i + 1] >= s[i] - tol for i in range(len(s) - 1))


def measure_gauntlet_rate(
    game_runner: GameRunner,
    opponent_kinds: list[str] | tuple[str, ...],
    *,
    games_per_opponent: int,
    seed: int = 0,
) -> float:
    """REUSE A5 ``play_gauntlet`` (``a_gate.py:753``) to play the promotion-bench
    gauntlet and return the candidate's score rate. This is the gauntlet
    component's measurement entry point -- B6 composes A5 ``play_gauntlet`` (it
    does NOT rewrite the gauntlet runner). The wired ``game_runner`` plays the
    candidate vs ``opponent_kinds``; tests inject a fake runner (no MLX/Rust)."""
    outcomes = play_gauntlet(
        game_runner,
        opponent_kinds,
        games_per_opponent=games_per_opponent,
        seed=seed,
    )
    return outcomes.score_rate()


def evaluate_block_b_gate(
    *,
    h2h_rate: float,
    gauntlet_rate: float,
    mana_draw_rate: float,
    baseline: ManaDrawBaseline,
    p1_p2_gap: float,
    aggregate_history: list[float] | tuple[float, ...],
    n_snap: int = DEFAULT_BLOCK_B_N_SNAP,
    mana_draw_band_low: float = MANA_DRAW_BAND_LOW,
    mana_draw_band_high: float = MANA_DRAW_BAND_HIGH,
    current_hand_cap: int | None = None,
    current_mana_draw_base: int | None = None,
    h2h_threshold: float = BLOCK_B_H2H_MIN_SCORE_RATE,
    gauntlet_threshold: float = BLOCK_B_GAUNTLET_MIN_SCORE_RATE,
    p1_p2_gap_threshold: float = P1_P2_GAP_THRESHOLD,
    monotone_tolerance: float = DEFAULT_MONOTONE_TOLERANCE,
    internal_metrics: CandidateInternalMetrics | None = None,
) -> BlockBGateResult:
    """Evaluate the 4 Block-B external-bench components + the monotone-aggregate
    promotion verdict (``design.md:121``).

    Builds the 4 ``GateOutcome`` s:
      * ``h2h_vs_best`` -- records ``h2h_rate``; passes iff ``>= h2h_threshold``.
      * ``gauntlet`` -- records ``gauntlet_rate``; passes iff ``>= gauntlet_threshold``.
      * ``mana_draw_band`` -- REUSES A5 ``check_mana_draw_band`` (``a_gate.py:309``)
        with ``mana_draw_rate`` + ``baseline`` + the band/Q4-guard kwargs.
      * ``p1_p2_gap`` -- records ``p1_p2_gap``; passes iff ``<= p1_p2_gap_threshold``.

    Computes the current ``block_b_aggregate`` and appends it to a copy of
    ``aggregate_history`` (the prior snapshots' aggregates, oldest-first) to form
    ``monotone_aggregate_history``. The promotion verdict ``passed`` is True iff:
      1. ``len(monotone_aggregate_history) >= n_snap`` (enough snapshots), AND
      2. the last ``n_snap`` aggregates are non-decreasing within
         ``monotone_tolerance`` (monotone improvement of the FULL aggregate), AND
      3. all 4 components pass for the most-recent snapshot.

    First-snapshot / insufficient-snapshots case (``len < n_snap``): ``passed`` is
    False with ``reason == 'insufficient_snapshots'`` -- the caller seeds the
    best-ever anchor (A5 ``current_best_h2h_score_rate=None`` pattern,
    ``a_gate.py:632``); no plateau is declared yet.

    Promotion-by-loss guard (INHERITED from A5 ``select_promotion`` ``a_gate.py:
    638-641``): ``internal_metrics`` is accepted but NEVER read -- the deliberate
    absence of any read of ``internal_metrics.ppo_loss`` / ``approx_kl`` /
    ``entropy`` below IS the guard. Two candidates with identical external-bench
    but different internal metrics get the SAME decision.
    """
    # --- the 4 components (mana_draw_band REUSES A5 check_mana_draw_band) -------
    h2h_outcome = _check_h2h_vs_best(h2h_rate, threshold=h2h_threshold)
    gauntlet_outcome = _check_gauntlet(gauntlet_rate, threshold=gauntlet_threshold)
    mana_outcome = check_mana_draw_band(
        mana_draw_rate,
        baseline,
        current_hand_cap=current_hand_cap,
        current_mana_draw_base=current_mana_draw_base,
        band_low=mana_draw_band_low,
        band_high=mana_draw_band_high,
    )
    gap_outcome = _check_p1_p2_gap(p1_p2_gap, threshold=p1_p2_gap_threshold)

    # --- current aggregate + append to history ---------------------------------
    current_aggregate = block_b_aggregate(
        h2h_rate=h2h_rate,
        gauntlet_rate=gauntlet_rate,
        mana_draw_in_band=mana_outcome.passed,
        p1_p2_gap=p1_p2_gap,
    )
    history = [float(x) for x in aggregate_history] + [current_aggregate]
    history_tuple = tuple(history)

    all_four_pass = (
        h2h_outcome.passed
        and gauntlet_outcome.passed
        and mana_outcome.passed
        and gap_outcome.passed
    )

    n = int(n_snap)
    if n <= 0:
        raise ValueError("n_snap must be positive")

    insufficient = len(history) < n
    if insufficient:
        # First-snapshot / seed-anchor case: no promote yet, no plateau. The
        # caller seeds the best-ever anchor (A5 ``current_best_h2h_score_rate=
        # None`` pattern, ``a_gate.py:632``). ``passed`` is False; the
        # ``reason`` discriminator carries ``'insufficient_snapshots'`` so a
        # caller can tell this not-yet-promote apart from a monotone/component
        # failure without re-deriving from history length + failed_criteria().
        passed = False
        reason = "insufficient_snapshots"
    else:
        recent = history[-n:]
        monotone = _monotone_non_decreasing(recent, tolerance=monotone_tolerance)
        passed = bool(monotone and all_four_pass)
        if passed:
            reason = "promoted"
        elif not all_four_pass:
            reason = "component_failed"
        else:
            reason = "monotone_not_improving"

    # GUARD: promotion is EXTERNAL-BENCH ONLY. ``internal_metrics`` is NOT
    # consulted anywhere below (or above). (The deliberate absence of any read of
    # ``internal_metrics.ppo_loss`` / ``internal_metrics.approx_kl`` /
    # ``internal_metrics.entropy`` IS the inherited A5 promotion-by-loss guard,
    # ``a_gate.py:638-641``. A regression test asserts two candidates with
    # identical external-bench but different internal metrics get the same
    # decision -- see test_promotion_independent_of_ppo_loss.)
    _ = internal_metrics  # accepted, deliberately unused (monitoring-only)

    return BlockBGateResult(
        h2h_vs_best=h2h_outcome,
        gauntlet=gauntlet_outcome,
        mana_draw_band=mana_outcome,
        p1_p2_gap=gap_outcome,
        passed=bool(passed),
        monotone_aggregate_history=history_tuple,
        n_snap=n,
        reason=reason,
    )


__all__ = [
    "BLOCK_B_GAUNTLET_MIN_SCORE_RATE",
    "BLOCK_B_H2H_MIN_SCORE_RATE",
    "BlockBGateResult",
    "DEFAULT_BLOCK_B_N_SNAP",
    "DEFAULT_MONOTONE_TOLERANCE",
    "P1_P2_GAP_THRESHOLD",
    "block_b_aggregate",
    "evaluate_block_b_gate",
    "measure_gauntlet_rate",
]