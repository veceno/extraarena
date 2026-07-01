"""Block B component B7 -- ``exit_to_c2.py`` -- plateau-below-dominance-target
exit signal to C2 (the human-data collection phase).

Purpose (``BLOCK_B_PLAN.md:577-595`` / ``design.md:122,125``): detect that the
H2H-vs-best-self-snapshot series has PLATEAUED (no gain > ``min_gain`` over
``K_snap`` consecutive snapshots) AND the candidate is still BELOW the dominance
target -- i.e. the league is stuck below dominance and genuine further self-play
improvement has stalled, so the next lever is C2 (deploy the best V5 vs humans in
``rlhf_env``, collect ~3-5k fresh preV5-vs-human battles, ``design.md:125``).

This is the INVERSE of A5 ``check_h2h_trending`` (``a_gate.py:374-434``): A5
detects the series TRENDING UP (the most recent ``min_snapshots`` are
non-decreasing within tolerance -- improvement is happening). B7 detects the
series NOT trending up by more than ``min_gain`` over ``K_snap`` snapshots -- the
PLATEAU. Source-vs-source: a series A5 flags as trending-up should NOT plateau in
B7, and a flat series A5 flags as NOT-trending SHOULD plateau in B7.

D-B3 (CONFIRMED 2026-07-01) = BELOW + progression: the exit fires when the
plateau is BELOW the dominance target (still-weak -> C2 human data breaks the
plateau). A plateau AT/ABOVE the dominance target (dominant) does NOT fire this
exit (dominant plateau -> Block E1 ship path, NOT C2). The default reading is
``below_target_exits=True`` (the D-B3 below reading). B7 is STILL FLIPPABLE: set
``below_target_exits=False`` to express the at/above reading (D-B3 3a revisited):
then a plateau AT/ABOVE the target fires the exit and a plateau BELOW the target
does NOT.

D-B2 defaults: ``K_snap`` ~10 (~2x ``N_snap``=5, ``a_gate.py`` / Block-B
``DEFAULT_BLOCK_B_N_SNAP``=5), ``min_gain``=0.01 H2H score-rate (implementer
default -- small enough that genuine improvement clears it while noise does not;
NOT a user decision).

B6/B1 grounding: B6 ``BlockBGateResult.h2h_vs_best`` (``block_b_gate.py``, the
``GateOutcome`` H2H-vs-best series source). B1 ``SnapshotPool.best_ever``
(``snapshot_pool.py:181``, the best-ever anchor ``path`` = the C2 deploy
candidate checkpoint). For synthetic tests B7 takes the H2H series +
``best_checkpoint_path`` as fabricated inputs. B7 does NOT depend on B8.

frozen-classic guard: no edit to ``classic_*`` / ``reward_v5`` / ``v5_trace`` /
``warm_start_v5`` / ``run_phase26*`` / ``run_v5_acceptance`` / ``league_v5.py`` /
``gauntlet_v5.py`` / ``opponents_v5.py`` (read-only). NO edit to A5 ``a_gate.py``
(``check_h2h_trending`` inverse-reference read-only; regression guard
``git diff a_gate.py`` empty). NO edit to B6 ``block_b_gate.py`` (consume
H2H-vs-best read-only). NO edit to B1 ``snapshot_pool.py`` (consume best-ever
path read-only). NO Rust edit. NO TrainV3.5-into-prod. Synthetic tests only
(fabricated H2H series, no real Rust/MLX/ONNX).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


#: D-B2 default -- number of consecutive snapshots over which the H2H-vs-best
#: series must show NO gain > ``min_gain`` to count as a plateau
#: (``BLOCK_B_PLAN.md:161``, ``design.md:122``). ~2x ``N_snap``=5.
DEFAULT_K_SNAP: int = 10

#: D-B2 default -- per-step gain tolerance (H2H score-rate). A per-step gain
#: <= ``min_gain`` counts as 'no gain' (the plateau run continues); a gain >
#: ``min_gain`` RESETS the run (genuine improvement). Small enough that genuine
#: improvement clears it while noise does not. Implementer default, NOT a user
#: decision (``BLOCK_B_PLAN.md:161``).
DEFAULT_MIN_GAIN: float = 0.01

#: D-B3 default reading -- exit fires when the plateau is BELOW the dominance
#: target (still-weak -> C2). Set False to flip to the at/above reading
#: (D-B3 3a revisited).
DEFAULT_BELOW_TARGET_EXITS: bool = True


@dataclass(frozen=True)
class ExitToC2Verdict:
    """The B7 exit-to-C2 verdict (frozen dataclass).

    ``exit_fires`` is the boolean exit signal (True -> emit the exit->C2 signal,
    deploy best V5 vs humans in ``rlhf_env``). ``plateau`` is True iff the H2H
    series showed no gain > ``min_gain`` over ``K_snap`` consecutive snapshots.
    ``below_target`` is True iff the most recent H2H score is below
    ``dominance_target``. ``plateau_run_length`` is the no-gain run length ending
    at the most recent snapshot (0 if the last step was a genuine gain).
    ``best_checkpoint_path`` is the B1 best-ever anchor path (the C2 deploy
    candidate checkpoint), carried unchanged from the caller.
    """

    exit_fires: bool
    plateau: bool
    below_target: bool
    current_h2h: float
    dominance_target: float
    k_snap: int
    min_gain: float
    plateau_run_length: int
    best_checkpoint_path: str | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the verdict to a plain dict (for logging / JSON export)."""
        return {
            "exit_fires": self.exit_fires,
            "plateau": self.plateau,
            "below_target": self.below_target,
            "current_h2h": self.current_h2h,
            "dominance_target": self.dominance_target,
            "k_snap": self.k_snap,
            "min_gain": self.min_gain,
            "plateau_run_length": self.plateau_run_length,
            "best_checkpoint_path": self.best_checkpoint_path,
            "reason": self.reason,
            "details": dict(self.details),
        }


def detect_h2h_plateau(
    h2h_scores: list[float] | tuple[float, ...],
    *,
    dominance_target: float,
    K_snap: int = DEFAULT_K_SNAP,
    min_gain: float = DEFAULT_MIN_GAIN,
    below_target_exits: bool = DEFAULT_BELOW_TARGET_EXITS,
    best_checkpoint_path: str | None = None,
) -> ExitToC2Verdict:
    """Detect an H2H-vs-best plateau below the dominance target -> exit to C2.

    ``h2h_scores`` is the series of measured H2H score rates (candidate vs the
    current best self-snapshot, one per snapshot, oldest-first) -- the same
    series A5 ``check_h2h_trending`` consumes (``a_gate.py:374``).

    Plateau run-length logic: iterate consecutive pairs; a per-step gain
    (``scores[i+1] - scores[i]``) <= ``min_gain`` counts as 'no gain' (run++);
    a gain > ``min_gain`` RESETS the run to 0. ``plateau`` is True iff the run
    ending at the most recent snapshot is >= ``K_snap``. (A single flat
    snapshot -> run=1 < ``K_snap`` -> not plateau; a sub-``min_gain`` uptick
    (+0.005) -> no-gain -> run continues; an above-``min_gain`` uptick (+0.02)
    -> resets.)

    Insufficient data: if ``len(scores) < K_snap + 1`` (need ``K_snap``
    consecutive no-gain steps, i.e. ``K_snap + 1`` points) -> ``exit_fires``
    False, reason ``'insufficient_snapshots'``.

    DEFAULT reading (``below_target_exits=True``, D-B3 below+progression):
    ``exit_fires = plateau AND below_target``. A plateau AT/ABOVE the target
    -> ``exit_fires`` False (reason ``'dominant_plateau_e1_path'`` -- dominant
    plateau goes to Block E1 ship, NOT C2). FLIPPED reading
    (``below_target_exits=False``, D-B3 3a at/above): ``exit_fires = plateau AND
    (not below_target)``; a plateau BELOW the target -> ``exit_fires`` False
    (reason ``'still_improving_below'`` -- keep training).

    A still-improving series (gains > ``min_gain``) -> run resets, never reaches
    ``K_snap`` -> ``plateau`` False -> ``exit_fires`` False (reason
    ``'still_improving'``).

    ``best_checkpoint_path`` (the B1 best-ever anchor path, the C2 deploy
    candidate) is carried unchanged on the verdict.
    """
    scores = [float(s) for s in h2h_scores]
    k = int(K_snap)
    gain_tol = float(min_gain)
    target = float(dominance_target)
    if k <= 0:
        raise ValueError("K_snap must be positive")
    if gain_tol < 0.0:
        raise ValueError("min_gain must be non-negative")

    current_h2h = scores[-1] if scores else 0.0
    below_target = current_h2h < target

    # Insufficient data: need K_snap consecutive no-gain steps -> K_snap+1 points.
    if len(scores) < k + 1:
        return ExitToC2Verdict(
            exit_fires=False,
            plateau=False,
            below_target=below_target,
            current_h2h=current_h2h,
            dominance_target=target,
            k_snap=k,
            min_gain=gain_tol,
            plateau_run_length=0,
            best_checkpoint_path=best_checkpoint_path,
            reason="insufficient_snapshots",
            details={
                "criterion": "h2h_vs_best_plateau_below_dominance",
                "n_measured": len(scores),
                "k_snap": k,
                "needed": k + 1,
                "spec_source": "design.md:122",
            },
        )

    # Plateau run-length: track the no-gain run ending at the most recent
    # snapshot. A per-step gain <= min_gain -> run++; gain > min_gain -> reset.
    run = 0
    for i in range(len(scores) - 1):
        gain = scores[i + 1] - scores[i]
        if gain > gain_tol:
            run = 0
        else:
            run += 1
    plateau = run >= k

    if not plateau:
        return ExitToC2Verdict(
            exit_fires=False,
            plateau=False,
            below_target=below_target,
            current_h2h=current_h2h,
            dominance_target=target,
            k_snap=k,
            min_gain=gain_tol,
            plateau_run_length=run,
            best_checkpoint_path=best_checkpoint_path,
            reason="still_improving",
            details={
                "criterion": "h2h_vs_best_plateau_below_dominance",
                "n_measured": len(scores),
                "k_snap": k,
                "min_gain": gain_tol,
                "plateau_run_length": run,
                "spec_source": "design.md:122",
            },
        )

    # Plateau confirmed. Apply the D-B3 reading.
    if below_target_exits:
        # DEFAULT (D-B3 below+progression): exit fires when plateau AND below.
        exit_fires = plateau and below_target
        if exit_fires:
            reason = "plateau_below_dominance_target"
        else:
            # Plateau AT/ABOVE target -> dominant -> E1 ship path, NOT C2.
            reason = "dominant_plateau_e1_path"
    else:
        # FLIPPED (D-B3 3a at/above): exit fires when plateau AND at/above.
        exit_fires = plateau and (not below_target)
        if exit_fires:
            reason = "plateau_at_or_above_dominance_target"
        else:
            # Plateau below target -> keep training (still-improving-below path).
            reason = "still_improving_below"

    return ExitToC2Verdict(
        exit_fires=exit_fires,
        plateau=True,
        below_target=below_target,
        current_h2h=current_h2h,
        dominance_target=target,
        k_snap=k,
        min_gain=gain_tol,
        plateau_run_length=run,
        best_checkpoint_path=best_checkpoint_path,
        reason=reason,
        details={
            "criterion": "h2h_vs_best_plateau_below_dominance",
            "n_measured": len(scores),
            "k_snap": k,
            "min_gain": gain_tol,
            "plateau_run_length": run,
            "below_target_exits": below_target_exits,
            "spec_source": "design.md:122",
        },
    )