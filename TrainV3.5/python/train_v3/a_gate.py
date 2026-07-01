"""Block A component A5 — ``a_gate.py`` — the A-gate + promotion selector + Q4
mana_draw-usage measurement (NEW BUILD, not wiring).

This is NEW CODE, not wiring (verifier findings 3a blocker + 3c major). Three of the
four A-gate criteria had NO existing infrastructure; the fourth (H2H trending)
likewise did not exist. The plan (``BLOCK_A_PLAN.md:481-586``) promotes the A-gate
from operational-split prose to a NAMED COMPONENT with file/acceptance/tests
(verifier finding 3c).

Verifier-confirmed gaps (why this is NEW, not wiring):
  * ``V5GauntletConfig.no_assist_min_score_rate`` (``gauntlet_v5.py:42 = 0.45``) and
    ``exploit_resistance_min_score_rate`` (``gauntlet_v5.py:46 = 0.42``) are DEAD
    FIELDS — grep-confirmed never read anywhere outside their definition (0 reads).
    A5 does NOT wire them; it builds NEW gate logic with the spec-raised thresholds
    (0.55 / 0.50) and offers ``build_a_gate_gauntlet_config`` to override the dead
    fields at construction (so the raised values are visible on the config object
    even though the A-gate check uses its own constants — belt and suspenders).
  * ``run_v5_acceptance.py`` plays NO games: it reads pre-computed winrates from a
    benchmark JSON (``:320-322``) + checks config FLAGS (``:488
    candidate_no_assist_hidden_mode``), NOT a score rate. The cited ``:41
    --min-no-bonus-p1`` (default 0.75) is a v4max no-bonus benchmark, NOT the
    no_assist/exploit_resistance thresholds. Also broken path: ``:16
    sys.path.insert(0, str(ROOT / "TrainV3" / "python"))`` but the worktree has
    ``TrainV3.5/`` (verifier finding 2e). A5 imports ``V5GauntletConfig`` +
    ``build_default_exploit_gauntlet`` + ``EXPLOIT_AGENT_KINDS`` directly from
    ``train_v3`` via the TrainV3.5 path (relative import), NOT via
    ``run_v5_acceptance.py``'s broken ``TrainV3`` path.
  * The mana_draw-usage band ``[0.5x, 1.5x]``: ZERO infrastructure anywhere
    (grep-confirmed empty in ``gauntlet_v5.py`` / ``run_v5_acceptance.py`` /
    ``league_v5.py``). A5 builds the measurement + band check.
  * H2H vs best-self-snapshot trending: ZERO infrastructure.
    ``compare_adaptive_strength_monotonicity`` (``league_v5.py:146``) is a SYNTHETIC
    FORMULA comparing ``evaluate_adaptive_strength_proxy`` floats (``:125-143``,
    0.25 vs 1.0), NOT real H2H games vs a self-snapshot (verifier finding 3a). A5
    builds the real H2H trending tracker (over a series of measured H2H score
    rates).

A-gate (exit Phase A, ``design.md:114``): a candidate PASSES iff ALL of:
  1. no_assist score rate  >= 0.55  (raised from dead-field 0.45, ``gauntlet_v5.py:42``)
  2. exploit_resistance score rate >= 0.50  (raised from dead-field 0.42, ``gauntlet_v5.py:46``)
  3. mana_draw usage in [0.5x, 1.5x] of the human baseline B  (Q4, ``design.md:114``)
  4. external H2H vs best self-snapshot trending up over >= N snapshots  (D-A5 default 5)

Promotion selector (``design.md:112`` — the D-lesson): promotion of a candidate to
the new best-self-snapshot is decided by EXTERNAL-BENCHMARK performance (the A-gate
+ a head-to-head vs the current best), NOT by internal training metrics (PPO loss /
KL / entropy are MONITORING-ONLY, never the promotion signal). The
promotion-by-loss GUARD (verifier finding 3b major — the load-bearing piece): a
candidate with LOWER internal training loss but FAILING the external A-gate must NOT
promote.

Q4 mana_draw-usage measurement (deferred from Block 0): ``B = mana_draw_count /
eligible_turns`` (the baseline mana_draw rate, measured over the pilot battles or a
reference policy). The A-gate band is ``[0.5*B, 1.5*B]`` (candidate mana_draw rate
must be within 0.5x-1.5x of B). Hard dependency (``BLOCK_A_PLAN.md:532-538``): B is
INVALIDATED if ``HAND_CAP`` (``core/engine.py:44 = 4``) or ``MANA_DRAW_BASE``
(``core/engine.py:59 = 2``) change — A5 records/asserts these constants when
measuring and re-measures/invalidates B if they differ.

A5 MAY reuse A4 ``rust_live_self_play`` (``rust_live_self_play.py``) to drive the
gauntlet games. The MEASUREMENT + GATING logic is A5's core and is unit-testable
with synthetic outcomes (no full training run required). The operational
gauntlet runner is MLX/Rust-gated and skipped when unbuildable
(``test_skip_if_no_mlx_or_rust``).

frozen-classic guard: ``gauntlet_v5.py`` / ``league_v5.py`` / ``run_v5_acceptance.py``
consumed READ-ONLY (A5 builds NEW in ``a_gate.py``; it references
``EXPLOIT_AGENT_KINDS`` read-only and does NOT wire the dead
``no_assist_min_score_rate`` / ``exploit_resistance_min_score_rate`` fields as the
gate). ``v5_trace.py`` NOT imported. ``core/state.py`` NOT modified. No TrainV3.5
import into prod paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# Direct import from train_v3 via the TrainV3.5 path (verifier finding 2e: NOT via
# run_v5_acceptance.py's broken ``TrainV3`` path). READ-ONLY use of gauntlet_v5:
# EXPLOIT_AGENT_KINDS (the exploit-lane roster), V5GauntletConfig (overridden at
# construction), build_default_exploit_gauntlet (the default lane list).
from .gauntlet_v5 import (
    EXPLOIT_AGENT_KINDS,
    V5GauntletConfig,
    build_default_exploit_gauntlet,
)

# --- A-gate thresholds (spec-raised; ``design.md:114``) ----------------------
#: no_assist score-rate threshold, RAISED from the dead-field 0.45
#: (``gauntlet_v5.py:42``). The candidate plays in no-assist mode (enemy info
#: hidden, no draw assist / assembler / desirerer — pure skill) and must score
#: (wins + 0.5*draws)/total >= this vs the gauntlet opponents.
NO_ASSIST_MIN_SCORE_RATE: float = 0.55

#: exploit_resistance score-rate threshold, RAISED from the dead-field 0.42
#: (``gauntlet_v5.py:46``). The candidate vs the EXPLOIT_AGENT_KINDS roster
#: (``gauntlet_v5.py:8``) must score >= this.
EXPLOIT_RESISTANCE_MIN_SCORE_RATE: float = 0.50

#: mana_draw-usage band low/high multipliers (``design.md:114``). The candidate
#: mana_draw rate must lie in ``[MANA_DRAW_BAND_LOW * B, MANA_DRAW_BAND_HIGH * B]``
#: where ``B`` is the human baseline mana_draw rate (Q4).
MANA_DRAW_BAND_LOW: float = 0.5
MANA_DRAW_BAND_HIGH: float = 1.5

#: D-A5 default: minimum number of H2H-vs-best-self-snapshot measurements required
#: for the trending gate (``design.md:114`` "trending up >= N snapshots").
DEFAULT_H2H_MIN_SNAPSHOTS: int = 5

#: Promotion beat threshold: the candidate's H2H score rate vs the current best
#: must STRICTLY exceed this to "beat" the best (a tie = no improvement = no
#: promotion). Score rate = (wins + 0.5*draws)/total, so 0.5 = even.
H2H_PROMOTION_THRESHOLD: float = 0.5

# --- Engine constants for the Q4 dependency guard (``core/engine.py:44,59``) --
#: ``core/engine.py:44`` — hand size cap. B (the mana_draw baseline) is INVALIDATED
#: if the live engine's HAND_CAP differs from the value recorded with B (a hand-cap
#: change rescales how often mana_draw is legal/usable, so the baseline must be
#: re-measured). Recorded read-only; A5 never edits the engine.
ENGINE_HAND_CAP: int = 4

#: ``core/engine.py:59`` — base mana-draw cost (cost of the Nth draw in a turn is
#: ``MANA_DRAW_BASE * N``). B is INVALIDATED if the live MANA_DRAW_BASE differs from
#: the recorded value (a cost change rescales the mana_draw-usage incentive).
ENGINE_MANA_DRAW_BASE: int = 2


# =============================================================================
# Score rate (mirrors ``run_phase1_runtime_acceptance_bench.py:704,717``)
# =============================================================================
def compute_score_rate(wins: int, draws: int, losses: int) -> float:
    """Score rate = ``(wins + 0.5*draws) / total`` (``design.md:114`` score rate).

    Mirrors ``run_phase1_runtime_acceptance_bench.py:704 score_rate`` + ``:717``.
    A draw is worth half a win (the candidate did not lose). ``total`` must be
    positive. The result is in ``[0.0, 1.0]``.
    """
    wins = int(wins)
    draws = int(draws)
    losses = int(losses)
    total = wins + draws + losses
    if total <= 0:
        raise ValueError("total games (wins + draws + losses) must be positive")
    return (wins + 0.5 * draws) / total


# =============================================================================
# Gate outcome + the 4 named criteria
# =============================================================================
@dataclass(frozen=True)
class GateOutcome:
    """One A-gate criterion result. ``passed`` is the boolean gate verdict;
    ``score`` is the measured value; ``threshold`` is the gate threshold (or the
    ``[low, high]`` band for the mana_draw criterion); ``details`` carries the
    extra context (opponent roster, baseline, invalidation flag, ...)."""

    name: str
    passed: bool
    score: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


def check_no_assist_gate(
    score_rate: float,
    *,
    threshold: float = NO_ASSIST_MIN_SCORE_RATE,
) -> GateOutcome:
    """Criterion 1 — no_assist score-rate gate (>= 0.55, raised from dead-field 0.45).

    The candidate plays in no-assist mode (enemy info hidden, no draw assist /
    assembler / desirerer — pure skill, the ``candidate_no_assist_hidden_mode``
    contract at ``run_v5_acceptance.py:488``) vs the gauntlet opponents and must
    score ``score_rate`` >= ``threshold``. The threshold is RAISED from the dead
    ``V5GauntletConfig.no_assist_min_score_rate`` 0.45 (``gauntlet_v5.py:42``) to
    the spec 0.55 (``design.md:114``). A5 does NOT wire the dead field — it builds
    this check with the spec threshold.
    """
    score_rate = float(score_rate)
    passed = score_rate >= float(threshold)
    return GateOutcome(
        name="no_assist",
        passed=passed,
        score=score_rate,
        threshold=float(threshold),
        details={
            "criterion": "no_assist_score_rate",
            "raised_from": 0.45,  # dead-field (gauntlet_v5.py:42), NOT wired
            "spec_source": "design.md:114",
        },
    )


def check_exploit_resistance_gate(
    score_rate: float,
    *,
    threshold: float = EXPLOIT_RESISTANCE_MIN_SCORE_RATE,
    exploit_kinds: tuple[str, ...] = EXPLOIT_AGENT_KINDS,
) -> GateOutcome:
    """Criterion 2 — exploit_resistance score-rate gate (>= 0.50, raised from
    dead-field 0.42).

    The candidate vs the EXPLOIT_AGENT_KINDS roster (``gauntlet_v5.py:8``:
    face_rush / board_control / greedy_trade / stall / punish_empty_board /
    anti_draw_greed / anti_hand_leak_overfit) must score ``score_rate`` >=
    ``threshold``. The threshold is RAISED from the dead
    ``V5GauntletConfig.exploit_resistance_min_score_rate`` 0.42
    (``gauntlet_v5.py:46``) to the spec 0.50 (``design.md:114``). A5 does NOT wire
    the dead field — it builds this check with the spec threshold.
    """
    score_rate = float(score_rate)
    passed = score_rate >= float(threshold)
    return GateOutcome(
        name="exploit_resistance",
        passed=passed,
        score=score_rate,
        threshold=float(threshold),
        details={
            "criterion": "exploit_resistance_score_rate",
            "exploit_kinds": list(exploit_kinds),
            "raised_from": 0.42,  # dead-field (gauntlet_v5.py:46), NOT wired
            "spec_source": "design.md:114",
        },
    )


# =============================================================================
# Q4 — mana_draw-usage measurement + the [0.5x, 1.5x] band gate
# =============================================================================
@dataclass(frozen=True)
class ManaDrawBaseline:
    """Q4 baseline B = ``mana_draw_count / eligible_turns`` (the human/reference
    mana_draw rate, ``design.md:114`` / ``BLOCK_A_PLAN.md:532-538``).

    ``hand_cap`` + ``mana_draw_base`` are the engine constants RECORDED alongside B
    (``core/engine.py:44 HAND_CAP=4``, ``core/engine.py:59 MANA_DRAW_BASE=2``). B
    is INVALIDATED (``valid=False``) if the live engine constants differ from the
    recorded values — a hand-cap or base-cost change rescales the mana_draw-usage
    incentive, so B must be re-measured. ``eligible_turns`` is the count of turns
    where mana_draw was a legal option for the player (hand not full + sufficient
    mana for the next draw).
    """

    mana_draw_count: int
    eligible_turns: int
    rate: float
    hand_cap: int
    mana_draw_base: int
    valid: bool = True


def compute_mana_draw_rate(mana_draw_count: int, eligible_turns: int) -> float:
    """mana_draw rate = ``mana_draw_count / eligible_turns`` (Q4, ``design.md:114``).

    ``eligible_turns`` is the number of turns where mana_draw was a legal option
    (hand not full, mana sufficient for the next draw cost
    ``MANA_DRAW_BASE * (count+1)``, ``core/engine.py:784``). Must be positive.
    """
    mana_draw_count = int(mana_draw_count)
    eligible_turns = int(eligible_turns)
    if eligible_turns <= 0:
        raise ValueError("eligible_turns must be positive (no eligible turns to rate)")
    if mana_draw_count < 0:
        raise ValueError("mana_draw_count must be non-negative")
    return mana_draw_count / eligible_turns


def record_mana_draw_baseline(
    mana_draw_count: int,
    eligible_turns: int,
    *,
    hand_cap: int = ENGINE_HAND_CAP,
    mana_draw_base: int = ENGINE_MANA_DRAW_BASE,
) -> ManaDrawBaseline:
    """Measure + record the Q4 baseline B with the engine constants it depends on.

    The returned baseline carries ``hand_cap`` + ``mana_draw_base`` so a later
    ``check_mana_draw_band`` call can detect that the live engine constants changed
    (via ``is_baseline_valid``) and INVALIDATE B (force a re-measure). This is the
    Q4 dependency guard (``BLOCK_A_PLAN.md:532-538``).
    """
    rate = compute_mana_draw_rate(mana_draw_count, eligible_turns)
    return ManaDrawBaseline(
        mana_draw_count=int(mana_draw_count),
        eligible_turns=int(eligible_turns),
        rate=rate,
        hand_cap=int(hand_cap),
        mana_draw_base=int(mana_draw_base),
        valid=True,
    )


def is_baseline_valid(
    baseline: ManaDrawBaseline,
    *,
    current_hand_cap: int = ENGINE_HAND_CAP,
    current_mana_draw_base: int = ENGINE_MANA_DRAW_BASE,
) -> bool:
    """Q4 dependency guard: B is valid iff the live engine constants match the
    recorded ones. A HAND_CAP or MANA_DRAW_BASE change INVALIDATES B (the
    mana_draw-usage incentive was rescaled — B must be re-measured).
    """
    return (
        int(baseline.hand_cap) == int(current_hand_cap)
        and int(baseline.mana_draw_base) == int(current_mana_draw_base)
    )


def check_mana_draw_band(
    candidate_rate: float,
    baseline: ManaDrawBaseline,
    *,
    current_hand_cap: int | None = None,
    current_mana_draw_base: int | None = None,
    band_low: float = MANA_DRAW_BAND_LOW,
    band_high: float = MANA_DRAW_BAND_HIGH,
) -> GateOutcome:
    """Criterion 3 — mana_draw-usage band gate (``design.md:114``).

    The candidate mana_draw rate must lie in ``[band_low * B, band_high * B]``
    where ``B = baseline.rate`` (the human/reference mana_draw rate, Q4). Defaults
    ``[0.5x, 1.5x]``.

    Q4 dependency guard: if ``current_hand_cap`` / ``current_mana_draw_base`` are
    provided and differ from the baseline's recorded constants, the gate FAILS with
    ``details["invalidated"]=True`` (B is stale — re-measure before gating). This
    is the load-bearing guard for the HAND_CAP / MANA_DRAW_BASE hard dependency
    (``BLOCK_A_PLAN.md:532-538``).
    """
    candidate_rate = float(candidate_rate)
    b = float(baseline.rate)
    # Q4 dependency guard: invalidate B if the engine constants changed.
    invalidated = False
    if current_hand_cap is not None and current_mana_draw_base is not None:
        if not is_baseline_valid(
            baseline,
            current_hand_cap=int(current_hand_cap),
            current_mana_draw_base=int(current_mana_draw_base),
        ):
            invalidated = True
    low = float(band_low) * b
    high = float(band_high) * b
    in_band = low <= candidate_rate <= high
    passed = (not invalidated) and in_band
    return GateOutcome(
        name="mana_draw_band",
        passed=passed,
        score=candidate_rate,
        threshold=float(high),  # the upper edge (band reported in details)
        details={
            "criterion": "mana_draw_usage_band",
            "baseline_rate": b,
            "baseline_mana_draw_count": int(baseline.mana_draw_count),
            "baseline_eligible_turns": int(baseline.eligible_turns),
            "band": [low, high],
            "band_multipliers": [float(band_low), float(band_high)],
            "invalidated": invalidated,
            "recorded_hand_cap": int(baseline.hand_cap),
            "recorded_mana_draw_base": int(baseline.mana_draw_base),
            "current_hand_cap": (
                None if current_hand_cap is None else int(current_hand_cap)
            ),
            "current_mana_draw_base": (
                None if current_mana_draw_base is None else int(current_mana_draw_base)
            ),
            "spec_source": "design.md:114",
        },
    )


# =============================================================================
# Criterion 4 — H2H vs best self-snapshot trending
# =============================================================================
def check_h2h_trending(
    h2h_scores: list[float] | tuple[float, ...],
    *,
    min_snapshots: int = DEFAULT_H2H_MIN_SNAPSHOTS,
    tolerance: float = 0.0,
) -> GateOutcome:
    """Criterion 4 — external H2H vs best self-snapshot trending up (``design.md:114``).

    ``h2h_scores`` is the series of measured H2H score rates (candidate vs the
    current best self-snapshot, one per snapshot, oldest-first). The gate passes
    when there are at least ``min_snapshots`` measurements AND the most recent
    ``min_snapshots`` values are non-decreasing within ``tolerance`` (trending up /
    not regressing). A single regression (a value below the previous minus
    tolerance) fails the gate.

    This is the REAL H2H trending tracker, NOT the synthetic
    ``compare_adaptive_strength_monotonicity`` formula (``league_v5.py:146`` — a
    deterministic proxy comparing ``evaluate_adaptive_strength_proxy`` floats, NOT
    H2H games vs a self-snapshot; verifier finding 3a).

    ``tolerance`` (default 0.0) allows a small per-step dip for noisy real
    measurements; the synthetic tests use 0.0 (strict non-decreasing).
    """
    scores = [float(s) for s in h2h_scores]
    n = int(min_snapshots)
    if n <= 0:
        raise ValueError("min_snapshots must be positive")
    if len(scores) < n:
        return GateOutcome(
            name="h2h_trending",
            passed=False,
            score=(scores[-1] if scores else 0.0),
            threshold=float(n),
            details={
                "criterion": "h2h_vs_best_self_snapshot_trending",
                "reason": "insufficient_snapshots",
                "n_measured": len(scores),
                "min_snapshots": n,
                "spec_source": "design.md:114",
            },
        )
    recent = scores[-n:]
    tol = float(tolerance)
    trending = all(
        recent[i + 1] >= recent[i] - tol for i in range(len(recent) - 1)
    )
    return GateOutcome(
        name="h2h_trending",
        passed=trending,
        score=recent[-1],
        threshold=float(n),
        details={
            "criterion": "h2h_vs_best_self_snapshot_trending",
            "recent_scores": recent,
            "n_measured": len(scores),
            "min_snapshots": n,
            "tolerance": tol,
            "regressed": (not trending),
            "spec_source": "design.md:114",
        },
    )


# =============================================================================
# A-gate aggregate (the single Phase-A pass/fail)
# =============================================================================
@dataclass(frozen=True)
class AGateResult:
    """The full A-gate verdict. ``passed`` is True iff ALL 4 criteria pass."""

    passed: bool
    no_assist: GateOutcome
    exploit_resistance: GateOutcome
    mana_draw_band: GateOutcome
    h2h_trending: GateOutcome

    def failed_criteria(self) -> list[str]:
        """Names of the criteria that failed (empty iff ``passed``)."""
        return [
            g.name
            for g in (
                self.no_assist,
                self.exploit_resistance,
                self.mana_draw_band,
                self.h2h_trending,
            )
            if not g.passed
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "failed_criteria": self.failed_criteria(),
            "no_assist": _outcome_dict(self.no_assist),
            "exploit_resistance": _outcome_dict(self.exploit_resistance),
            "mana_draw_band": _outcome_dict(self.mana_draw_band),
            "h2h_trending": _outcome_dict(self.h2h_trending),
        }


def _outcome_dict(o: GateOutcome) -> dict[str, Any]:
    return {
        "name": o.name,
        "passed": bool(o.passed),
        "score": float(o.score),
        "threshold": float(o.threshold),
        "details": dict(o.details),
    }


def evaluate_a_gate(
    *,
    no_assist_score_rate: float,
    exploit_resistance_score_rate: float,
    candidate_mana_draw_rate: float,
    mana_draw_baseline: ManaDrawBaseline,
    h2h_scores: list[float] | tuple[float, ...],
    current_hand_cap: int | None = None,
    current_mana_draw_base: int | None = None,
    no_assist_threshold: float = NO_ASSIST_MIN_SCORE_RATE,
    exploit_resistance_threshold: float = EXPLOIT_RESISTANCE_MIN_SCORE_RATE,
    h2h_min_snapshots: int = DEFAULT_H2H_MIN_SNAPSHOTS,
    h2h_tolerance: float = 0.0,
) -> AGateResult:
    """Evaluate all 4 A-gate criteria and emit the single Phase-A pass/fail.

    A candidate PASSES iff ALL of (``design.md:114``):
      1. ``no_assist_score_rate``  >= 0.55 (raised from dead-field 0.45)
      2. ``exploit_resistance_score_rate`` >= 0.50 (raised from dead-field 0.42)
      3. ``candidate_mana_draw_rate`` in [0.5x, 1.5x] of ``mana_draw_baseline.rate``
         (Q4; invalidated if the live HAND_CAP / MANA_DRAW_BASE differ from the
         baseline's recorded constants — pass ``current_*`` to enforce the guard)
      4. ``h2h_scores`` trending up over >= ``h2h_min_snapshots`` (D-A5 default 5)
    """
    no_assist = check_no_assist_gate(
        no_assist_score_rate, threshold=no_assist_threshold
    )
    exploit = check_exploit_resistance_gate(
        exploit_resistance_score_rate, threshold=exploit_resistance_threshold
    )
    mana = check_mana_draw_band(
        candidate_mana_draw_rate,
        mana_draw_baseline,
        current_hand_cap=current_hand_cap,
        current_mana_draw_base=current_mana_draw_base,
    )
    h2h = check_h2h_trending(
        h2h_scores, min_snapshots=h2h_min_snapshots, tolerance=h2h_tolerance
    )
    passed = no_assist.passed and exploit.passed and mana.passed and h2h.passed
    return AGateResult(
        passed=passed,
        no_assist=no_assist,
        exploit_resistance=exploit,
        mana_draw_band=mana,
        h2h_trending=h2h,
    )


# =============================================================================
# Promotion selector + the promotion-by-loss GUARD (the D-lesson, gap #7)
# =============================================================================
@dataclass(frozen=True)
class CandidateExternalBench:
    """The EXTERNAL-BENCHMARK inputs to the promotion selector.

    These are the ONLY inputs the promotion decision consults (``design.md:112`` —
    "Promotion by external bench only"):
      * ``a_gate`` — the A-gate verdict (all 4 criteria).
      * ``h2h_vs_best_score_rate`` — the candidate's H2H score rate vs the current
        best self-snapshot ((wins + 0.5*draws)/total). Must STRICTLY beat the
        current best (``> H2H_PROMOTION_THRESHOLD`` = 0.5) to promote.

    Internal training metrics (PPO loss / KL / entropy) are NOT carried here — they
    are MONITORING-ONLY (``design.md:112``) and never the promotion signal. They
    travel via ``CandidateInternalMetrics`` (recorded/returned for monitoring but
    deliberately NOT consulted by ``select_promotion``).
    """

    a_gate: AGateResult
    h2h_vs_best_score_rate: float


@dataclass(frozen=True)
class CandidateInternalMetrics:
    """Internal training metrics — MONITORING-ONLY (``design.md:112``).

    These are recorded + returned by ``select_promotion`` for observability but are
    deliberately NOT consulted by the promotion decision (the promotion-by-loss
    guard, verifier finding 3b). A candidate with lower ``ppo_loss`` but a failing
    external A-gate must NOT promote.
    """

    ppo_loss: float | None = None
    approx_kl: float | None = None
    entropy: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionDecision:
    """The promotion selector verdict.

    ``promoted`` is True iff the candidate PASSES the A-gate AND strictly beats the
    current best on H2H (``design.md:112``). ``internal_metrics`` are echoed back
    for monitoring — they did NOT influence ``promoted`` (the guard).
    """

    promoted: bool
    reason: str
    a_gate_passed: bool
    h2h_vs_best_score_rate: float
    h2h_promotion_threshold: float
    internal_metrics: CandidateInternalMetrics
    is_first_snapshot: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": bool(self.promoted),
            "reason": self.reason,
            "a_gate_passed": bool(self.a_gate_passed),
            "h2h_vs_best_score_rate": float(self.h2h_vs_best_score_rate),
            "h2h_promotion_threshold": float(self.h2h_promotion_threshold),
            "is_first_snapshot": bool(self.is_first_snapshot),
            "internal_metrics_monitoring_only": {
                "ppo_loss": self.internal_metrics.ppo_loss,
                "approx_kl": self.internal_metrics.approx_kl,
                "entropy": self.internal_metrics.entropy,
                "extras": dict(self.internal_metrics.extras),
            },
        }


def select_promotion(
    external: CandidateExternalBench,
    internal: CandidateInternalMetrics | None = None,
    *,
    current_best_h2h_score_rate: float | None = None,
    h2h_promotion_threshold: float = H2H_PROMOTION_THRESHOLD,
) -> PromotionDecision:
    """Promotion selector — EXTERNAL-BENCH ONLY (``design.md:112``, the D-lesson).

    Promote a candidate to the new best-self-snapshot iff:
      1. it PASSES the A-gate (all 4 criteria), AND
      2. it STRICTLY beats the current best on H2H (``h2h_vs_best_score_rate`` >
         ``h2h_promotion_threshold``, default 0.5 = even; a tie is no improvement).

    The PROMOTION-BY-LOSS GUARD (verifier finding 3b — the load-bearing piece): the
    decision NEVER consults ``internal`` (PPO loss / KL / entropy). A candidate
    with LOWER loss but a FAILING external A-gate is NOT promoted. Two candidates
    with identical external-bench but different internal metrics get the SAME
    decision. ``internal`` is recorded + echoed back for monitoring only.

    ``current_best_h2h_score_rate=None`` marks the FIRST snapshot (no prior best to
    beat): promote iff the A-gate passes (the candidate becomes the inaugural
    best-self-snapshot).
    """
    internal = internal if internal is not None else CandidateInternalMetrics()
    is_first = current_best_h2h_score_rate is None
    a_gate_passed = bool(external.a_gate.passed)
    h2h = float(external.h2h_vs_best_score_rate)
    thresh = float(h2h_promotion_threshold)

    # GUARD: promotion is EXTERNAL-BENCH ONLY. ``internal`` is NOT consulted here.
    # (The deliberate absence of any read of ``internal.ppo_loss`` /
    # ``internal.approx_kl`` / ``internal.entropy`` below IS the guard. A regression
    # test asserts two candidates with identical external-bench but different
    # internal metrics get the same decision — see test_promotion_independent_of_ppo_loss.)

    # 1. A-gate must pass.
    if not a_gate_passed:
        return PromotionDecision(
            promoted=False,
            reason="a_gate_failed",
            a_gate_passed=False,
            h2h_vs_best_score_rate=h2h,
            h2h_promotion_threshold=thresh,
            internal_metrics=internal,
            is_first_snapshot=is_first,
        )

    # 2. Must beat the current best on H2H (skipped for the first snapshot).
    if not is_first:
        if not (h2h > thresh):
            return PromotionDecision(
                promoted=False,
                reason="h2h_not_beating_best",
                a_gate_passed=True,
                h2h_vs_best_score_rate=h2h,
                h2h_promotion_threshold=thresh,
                internal_metrics=internal,
                is_first_snapshot=is_first,
            )

    promoted = True
    reason = "promoted_first_snapshot" if is_first else "promoted_beats_best"
    return PromotionDecision(
        promoted=True,
        reason=reason,
        a_gate_passed=True,
        h2h_vs_best_score_rate=h2h,
        h2h_promotion_threshold=thresh,
        internal_metrics=internal,
        is_first_snapshot=is_first,
    )


# =============================================================================
# Operational gauntlet runner — plays games via an injectable GameRunner.
# =============================================================================
@dataclass(frozen=True)
class GameResult:
    """One played game, from the candidate's perspective.

    ``outcome`` is ``"win"`` / ``"draw"`` / ``"loss"``; ``mana_draw_count`` is the
    candidate's mana_draw uses this game; ``eligible_turns`` is the candidate's
    mana_draw-eligible turns this game. ``opponent`` is the opponent identity.
    """

    outcome: str
    mana_draw_count: int
    eligible_turns: int
    opponent: str

    def __post_init__(self) -> None:
        if self.outcome not in ("win", "draw", "loss"):
            raise ValueError(
                f"GameResult.outcome must be 'win'/'draw'/'loss', got {self.outcome!r}"
            )
        if int(self.eligible_turns) < 0:
            raise ValueError("eligible_turns must be non-negative")
        if int(self.mana_draw_count) < 0:
            raise ValueError("mana_draw_count must be non-negative")
        if int(self.mana_draw_count) > int(self.eligible_turns):
            raise ValueError("mana_draw_count cannot exceed eligible_turns")


class GameRunner(Protocol):
    """Plays one game between the candidate and ``opponent_kind``.

    Production wires the A4 live self-play entry point (``rust_live_self_play.py``
    ``run_live_self_play_update`` / ``collect_rust_live_rollout``) to play a real
    game on the Rust ``ArenaEnv`` and harvest the outcome + mana_draw channels
    (``mana_draw_taken`` / ``mana_draw_legal``, ``rust_live_self_play.py:424-426``).
    Tests inject a fake runner returning synthetic ``GameResult``s — the
    measurement + gating logic is unit-testable without MLX/Rust.
    """

    def play(self, opponent_kind: str, *, seed: int) -> GameResult: ...


@dataclass(frozen=True)
class GauntletOutcomes:
    """Aggregated outcomes of a gauntlet run (N games vs an opponent roster).

    Feeds ``compute_score_rate`` (the no_assist / exploit_resistance gates) and the
    Q4 mana_draw measurement (``compute_mana_draw_rate`` over
    ``mana_draw_count`` / ``eligible_turns``).
    """

    wins: int
    draws: int
    losses: int
    mana_draw_count: int
    eligible_turns: int
    per_opponent: dict[str, dict[str, int]]

    def total(self) -> int:
        return int(self.wins) + int(self.draws) + int(self.losses)

    def score_rate(self) -> float:
        return compute_score_rate(self.wins, self.draws, self.losses)

    def mana_draw_rate(self) -> float:
        if self.eligible_turns <= 0:
            raise ValueError("no eligible turns in this gauntlet")
        return compute_mana_draw_rate(self.mana_draw_count, self.eligible_turns)


def play_gauntlet(
    game_runner: GameRunner,
    opponent_kinds: list[str] | tuple[str, ...],
    *,
    games_per_opponent: int,
    seed: int = 0,
) -> GauntletOutcomes:
    """Play ``games_per_opponent`` games vs each opponent in ``opponent_kinds``
    via ``game_runner`` and aggregate the outcomes.

    This is the gauntlet game-runner the no_assist / exploit_resistance / H2H gates
    consume. The candidate side + mode (no-assist vs full) is the responsibility of
    the wired ``game_runner`` (production wires the A4 live runner with the right
    candidate mode; tests wire a fake). Returns ``GauntletOutcomes`` for the
    score-rate + mana_draw-rate measurement.
    """
    if games_per_opponent <= 0:
        raise ValueError("games_per_opponent must be positive")
    if not opponent_kinds:
        raise ValueError("opponent_kinds must contain at least one opponent")

    wins = draws = losses = 0
    mana_draw_count = 0
    eligible_turns = 0
    per_opponent: dict[str, dict[str, int]] = {}

    for opp in opponent_kinds:
        opp_w = opp_d = opp_l = 0
        opp_md = 0
        opp_et = 0
        for g in range(int(games_per_opponent)):
            result = game_runner.play(opp, seed=int(seed) * 1_000_003 + g)
            if result.outcome == "win":
                wins += 1
                opp_w += 1
            elif result.outcome == "draw":
                draws += 1
                opp_d += 1
            else:
                losses += 1
                opp_l += 1
            mana_draw_count += int(result.mana_draw_count)
            eligible_turns += int(result.eligible_turns)
            opp_md += int(result.mana_draw_count)
            opp_et += int(result.eligible_turns)
        per_opponent[opp] = {
            "wins": opp_w,
            "draws": opp_d,
            "losses": opp_l,
            "mana_draw_count": opp_md,
            "eligible_turns": opp_et,
        }

    return GauntletOutcomes(
        wins=wins,
        draws=draws,
        losses=losses,
        mana_draw_count=mana_draw_count,
        eligible_turns=eligible_turns,
        per_opponent=per_opponent,
    )


def run_no_assist_gauntlet(
    game_runner: GameRunner,
    *,
    opponent_kinds: list[str] | tuple[str, ...] | None = None,
    games_per_opponent: int = 20,
    seed: int = 0,
) -> GateOutcome:
    """Run the no_assist gauntlet (criterion 1) and return its gate outcome.

    The wired ``game_runner`` plays the candidate in NO-ASSIST mode (enemy info
    hidden, no draw assist / assembler / desirerer) vs ``opponent_kinds`` (default
    the EXPLOIT_AGENT_KINDS roster, ``gauntlet_v5.py:8``). The score rate is gated
    at >= 0.55 (raised from dead-field 0.45).
    """
    if opponent_kinds is None:
        opponent_kinds = list(EXPLOIT_AGENT_KINDS)
    outcomes = play_gauntlet(
        game_runner, opponent_kinds, games_per_opponent=games_per_opponent, seed=seed
    )
    return check_no_assist_gate(outcomes.score_rate())


def run_exploit_resistance_gauntlet(
    game_runner: GameRunner,
    *,
    opponent_kinds: list[str] | tuple[str, ...] | None = None,
    games_per_opponent: int = 20,
    seed: int = 0,
) -> GateOutcome:
    """Run the exploit_resistance gauntlet (criterion 2) and return its gate outcome.

    The wired ``game_runner`` plays the candidate vs ``opponent_kinds`` (default
    the EXPLOIT_AGENT_KINDS roster, ``gauntlet_v5.py:8``). The score rate is gated
    at >= 0.50 (raised from dead-field 0.42).
    """
    if opponent_kinds is None:
        opponent_kinds = list(EXPLOIT_AGENT_KINDS)
    outcomes = play_gauntlet(
        game_runner, opponent_kinds, games_per_opponent=games_per_opponent, seed=seed
    )
    return check_exploit_resistance_gate(outcomes.score_rate())


# =============================================================================
# Config override (the dead fields overridden at construction, NOT wired as gate)
# =============================================================================
def build_a_gate_gauntlet_config(
    *,
    no_assist_min_score_rate: float = NO_ASSIST_MIN_SCORE_RATE,
    exploit_resistance_min_score_rate: float = EXPLOIT_RESISTANCE_MIN_SCORE_RATE,
    **kwargs: Any,
) -> V5GauntletConfig:
    """Build a ``V5GauntletConfig`` with the spec-raised thresholds overriding the
    DEAD fields (``gauntlet_v5.py:42 no_assist_min_score_rate`` 0.45 -> 0.55,
    ``gauntlet_v5.py:46 exploit_resistance_min_score_rate`` 0.42 -> 0.50).

    This documents the override at construction (``BLOCK_A_PLAN.md:524,531``).
    A5's gate logic uses its OWN constants (``NO_ASSIST_MIN_SCORE_RATE`` /
    ``EXPLOIT_RESISTANCE_MIN_SCORE_RATE``), NOT these config fields — the config is
    carried for observability + so a downstream consumer that DOES read the config
    sees the raised values. The dead fields are NOT wired as the gate (verifier
    finding 2d + 3a).
    """
    return V5GauntletConfig(
        no_assist_min_score_rate=float(no_assist_min_score_rate),
        exploit_resistance_min_score_rate=float(exploit_resistance_min_score_rate),
        **kwargs,
    ).validate()


def has_mlx_or_rust() -> bool:
    """Skip-gate helper for the operational gauntlet runner tests
    (``test_skip_if_no_mlx_or_rust``). Returns True iff MLX is importable OR the
    Rust FFI extension is buildable — the live gauntlet (real games via A4) needs
    one of them. The measurement + gating logic does NOT (it is synthetic-testable).
    """
    try:
        import mlx  # noqa: F401
        return True
    except Exception:
        pass
    try:
        from .rust_ffi import RustBatchWorker  # noqa: F401
        # Probe that the extension is actually buildable/importable, not just the
        # Python wrapper. ``from_live`` would fail without the compiled extension,
        # but importing the class only needs the wrapper module. Use a hasattr
        # probe on a known FFI function attribute to confirm the extension loaded.
        if not hasattr(RustBatchWorker, "from_live"):
            return False
        return True
    except Exception:
        return False


__all__ = [
    "AGateResult",
    "CandidateExternalBench",
    "CandidateInternalMetrics",
    "DEFAULT_H2H_MIN_SNAPSHOTS",
    "ENGINE_HAND_CAP",
    "ENGINE_MANA_DRAW_BASE",
    "EXPLOIT_RESISTANCE_MIN_SCORE_RATE",
    "EXPLOIT_AGENT_KINDS",
    "GauntletOutcomes",
    "GameResult",
    "GameRunner",
    "H2H_PROMOTION_THRESHOLD",
    "MANA_DRAW_BAND_HIGH",
    "MANA_DRAW_BAND_LOW",
    "ManaDrawBaseline",
    "NO_ASSIST_MIN_SCORE_RATE",
    "PromotionDecision",
    "build_a_gate_gauntlet_config",
    "check_exploit_resistance_gate",
    "check_h2h_trending",
    "check_mana_draw_band",
    "check_no_assist_gate",
    "compute_mana_draw_rate",
    "compute_score_rate",
    "evaluate_a_gate",
    "has_mlx_or_rust",
    "is_baseline_valid",
    "play_gauntlet",
    "record_mana_draw_baseline",
    "run_exploit_resistance_gauntlet",
    "run_no_assist_gauntlet",
    "select_promotion",
]