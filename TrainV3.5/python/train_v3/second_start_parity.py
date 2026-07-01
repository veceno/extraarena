"""B5 — continuous second-start parity loop (closes A4's open loop).

design.md:120 ``p1_p2_score_gap <= 0.12`` acceptance; oversample p2-init on
breach. The MECHANISM already exists — A3 ``second_start_oversampling_scheme``
(``ppo_phaseA_config.second_start_oversampling_scheme``) implements the gap-
weighted p1/p2 split, and A4 ``sample_learner_sides`` applies it — but A4 accepts
``p1_score_rate``/``p2_score_rate`` as INPUTS and never MEASURES them.

B5 CLOSES THE LOOP: it measures p1/p2 score rates over a rolling window of
side-stratified gauntlet games (the candidate played from BOTH p1 and p2), then
feeds the measured rates back into A3's ``second_start_oversampling_scheme``
(REUSED — B5 does NOT re-invent the gap-weight math) and exposes the gap as a
promotion input to B6 (B5 PRODUCES the gap; B6 CONSUMES it — B5 does NOT depend
on B6).

This module builds NEW measurement (A5 pattern — the dead
``gauntlet_v5.V5GauntletConfig.p1_p2_max_score_gap=0.12`` field has ZERO
consumers and is NOT wired here). It COMPOSES A5 ``GameResult``
(``a_gate.GameResult``, a frozen dataclass) inside a new ``BlockBGameResult``
rather than mutating the A5 dataclass — A5 ``GameResult`` is left unchanged.

Sources:
- A3 ``ppo_phaseA_config.second_start_oversampling_scheme`` (:258-305) — REUSED.
- A5 ``a_gate.GameResult`` (:685), ``GameRunner`` Protocol (:711),
  ``play_gauntlet`` (:753-813) — COMPOSED / mirrored side-stratified.
- A4 ``rust_live_self_play.sample_learner_sides`` (:495-517) — B5 feeds the
  measured rates here via the B8 driver; B5 does NOT edit the sampler.

Run: ``PYTHONPATH=.:TrainV3.5/python python3 -m pytest
TrainV3.5/python/train_v3/tests/test_second_start_parity.py``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

from train_v3.a_gate import GameResult
from train_v3.ppo_phaseA_config import (
    PHASE_A_P1_P2_GAP_THRESHOLD,
    second_start_oversampling_scheme,
)

# Block B does NOT import ``gauntlet_v5`` — the dead
# ``gauntlet_v5.V5GauntletConfig.p1_p2_max_score_gap`` field has ZERO consumers and
# B5 builds NEW measurement (A5 pattern). See test_does_not_wire_dead_field.
__all__ = [
    "BLOCK_B_GAP_THRESHOLD",
    "BlockBGameResult",
    "BlockBGameRunner",
    "SecondStartParityLoop",
    "play_side_stratified_gauntlet",
]

# The acceptance threshold (design.md:120 ``p1_p2_score_gap <= 0.12``). B5 reuses
# the same constant A3 uses (``PHASE_A_P1_P2_GAP_THRESHOLD``) so the breach
# definition is single-sourced; B5 does NOT re-invent the threshold.
BLOCK_B_GAP_THRESHOLD: float = PHASE_A_P1_P2_GAP_THRESHOLD

_VALID_SIDES = ("p1", "p2")


@dataclass(frozen=True)
class BlockBGameResult:
    """One played gauntlet game plus the candidate's starting side.

    COMPOSES A5 ``GameResult`` (``a_gate.GameResult``, a frozen dataclass) without
    mutating or duplicating its fields — the A5 ``GameResult`` is wrapped as the
    ``game`` field and is left unchanged. ``candidate_side`` is the side the
    candidate started from this game ("p1" or "p2"), which A5 ``GameResult`` does
    NOT record. Additive over A5 (A5 ``GameResult`` is unchanged — verify with
    ``git diff TrainV3.5/python/train_v3/a_gate.py`` empty).
    """

    game: GameResult
    candidate_side: str

    def __post_init__(self) -> None:
        if self.candidate_side not in _VALID_SIDES:
            raise ValueError(
                f"BlockBGameResult.candidate_side must be one of {_VALID_SIDES!r}, "
                f"got {self.candidate_side!r}"
            )
        # Validate the composed A5 GameResult is exactly the A5 type (composition,
        # not a duck-typed reimplementation) — guards against accidentally
        # wrapping a plain dict or re-defining GameResult locally.
        if not isinstance(self.game, GameResult):
            raise TypeError(
                "BlockBGameResult.game must be an a_gate.GameResult instance "
                f"(composition), got {type(self.game).__name__}"
            )


class BlockBGameRunner(Protocol):
    """Plays one game between the candidate and ``opponent_kind`` from a fixed
    candidate starting side.

    Extends A5 ``GameRunner`` (`a_gate.GameRunner`, :711) ADDITIVELY: A5
    ``play(opponent_kind, *, seed)`` has no side arg (the wired runner decides the
    candidate side internally); B5 adds an explicit ``candidate_side`` argument so
    the side-stratified gauntlet can play the candidate from BOTH p1 and p2.
    Production (B8) wires an A4-live-runner adapter that plays one real game on
    the Rust ``ArenaEnv`` with the candidate fixed to ``candidate_side``; tests
    inject a fake runner returning synthetic ``BlockBGameResult``s (no
    MLX/Rust/ONNX).
    """

    def play(
        self, opponent_kind: str, *, seed: int, candidate_side: str
    ) -> BlockBGameResult: ...


def play_side_stratified_gauntlet(
    runner: BlockBGameRunner,
    opponent_kinds: list[str] | tuple[str, ...],
    *,
    games_per_opponent_per_side: int,
    seed: int = 0,
) -> list[BlockBGameResult]:
    """Play a side-stratified gauntlet: ``games_per_opponent_per_side`` games vs
    each opponent in ``opponent_kinds`` from EACH candidate side (p1 and p2).

    Mirrors A5 ``play_gauntlet`` (`a_gate.play_gauntlet`, :753-813) in aggregation
    structure — same per-opponent loop, same seed derivation
    ``seed * 1_000_003 + g`` — but plays each opponent from BOTH p1 and p2 (the
    candidate side is explicit per game via ``BlockBGameRunner.play``) and returns
    the flat list of side-stratified ``BlockBGameResult``s (rather than an
    aggregated ``GauntletOutcomes``). The returned list feeds
    ``SecondStartParityLoop.update``.

    The candidate side + mode (no-assist vs full) is the responsibility of the
    wired ``runner`` (production wires the A4 live runner with the right candidate
    mode; tests wire a fake) — A5 ``play_gauntlet`` leaves the same responsibility
    to its wired ``game_runner``.
    """
    if games_per_opponent_per_side <= 0:
        raise ValueError("games_per_opponent_per_side must be positive")
    if not opponent_kinds:
        raise ValueError("opponent_kinds must contain at least one opponent")

    results: list[BlockBGameResult] = []
    for opp in opponent_kinds:
        for side in _VALID_SIDES:
            for g in range(int(games_per_opponent_per_side)):
                result = runner.play(
                    opp,
                    seed=int(seed) * 1_000_003 + g,
                    candidate_side=side,
                )
                results.append(result)
    return results


class SecondStartParityLoop:
    """Rolling-window p1/p2 second-start parity measurement + A3 scheme feedback.

    Maintains a ``deque(maxlen=window_n)`` of ``BlockBGameResult``s. ``update``
    appends recent side-stratified gauntlet results; stale games age out via the
    deque maxlen. ``p1_score_rate`` / ``p2_score_rate`` measure the candidate's
    score rate (``compute_score_rate`` semantics: ``(wins + 0.5*draws) / total``)
    SEPARATELY for games where ``candidate_side == "p1"`` / ``"p2"`` — NOT a single
    aggregate assuming a fixed side. ``gap`` = ``abs(p1_rate - p2_rate)``;
    ``breach`` = ``gap > 0.12``.

    ``oversampling_scheme`` REUSES A3 ``second_start_oversampling_scheme`` (the
    gap-weighted p1/p2 split) with the MEASURED rates — B5 does NOT re-invent the
    gap-weight math. On breach the scheme oversamples the LOWER-rate side (p2 if
    ``p2_rate < p1_rate``). ``gap_for_promotion`` exposes the measured gap to B6
    (continuous parity is a promotion criterion, design.md:121).

    An EMPTY side (no games recorded for that side) is treated as NEUTRAL: rate
    ``0.5``, contributing gap ``0`` and no breach — so a freshly-started loop with
    only p1 games does NOT spuriously breach on the missing p2 side.
    """

    def __init__(self, window_n: int, *, gap_threshold: float = BLOCK_B_GAP_THRESHOLD) -> None:
        if window_n <= 0:
            raise ValueError("window_n must be positive")
        self._results: deque[BlockBGameResult] = deque(maxlen=int(window_n))
        self._gap_threshold = float(gap_threshold)

    # -- accumulation -------------------------------------------------------
    def update(self, results: list[BlockBGameResult] | tuple[BlockBGameResult, ...]) -> None:
        """Append a batch of side-stratified gauntlet results to the rolling window.

        Results beyond ``window_n`` age out FIFO (deque maxlen) — only recent
        gauntlet games inform the rate (``test_rolling_window``).
        """
        for r in results:
            if not isinstance(r, BlockBGameResult):
                raise TypeError(
                    f"SecondStartParityLoop.update expects BlockBGameResult, got "
                    f"{type(r).__name__}"
                )
            self._results.append(r)

    # -- per-side measurement ----------------------------------------------
    def _side_stats(self, side: str) -> tuple[int, int, int]:
        wins = draws = losses = 0
        for r in self._results:
            if r.candidate_side != side:
                continue
            if r.game.outcome == "win":
                wins += 1
            elif r.game.outcome == "draw":
                draws += 1
            else:  # "loss"
                losses += 1
        return wins, draws, losses

    def p1_score_rate(self) -> float:
        """Candidate score rate when starting as p1:
        ``(wins_p1 + 0.5*draws_p1) / total_p1``. NEUTRAL 0.5 if no p1 games yet.
        """
        return self._side_rate("p1")

    def p2_score_rate(self) -> float:
        """Candidate score rate when starting as p2:
        ``(wins_p2 + 0.5*draws_p2) / total_p2``. NEUTRAL 0.5 if no p2 games yet.
        """
        return self._side_rate("p2")

    def _side_rate(self, side: str) -> float:
        wins, draws, losses = self._side_stats(side)
        total = wins + draws + losses
        if total <= 0:
            return 0.5  # neutral — no breach on a missing side
        return (wins + 0.5 * draws) / float(total)

    # -- gap / breach / promotion ------------------------------------------
    def gap(self) -> float:
        """``abs(p1_score_rate - p2_score_rate)`` over the rolling window."""
        return abs(self.p1_score_rate() - self.p2_score_rate())

    def breach(self) -> bool:
        """``gap > 0.12`` (``PHASE_A_P1_P2_GAP_THRESHOLD``) — oversample next update."""
        return self.gap() > self._gap_threshold

    def gap_for_promotion(self) -> float:
        """The measured p1/p2 score gap, exposed to B6 as a promotion input
        (continuous parity is a promotion criterion, design.md:121). B5 PRODUCES
        the gap; B6 CONSUMES it (B5 does NOT depend on B6).
        """
        return self.gap()

    def oversampling_scheme(self) -> dict:
        """REUSE A3 ``second_start_oversampling_scheme`` with the MEASURED rates.

        Returns ``{p1_weight, p2_weight, gap, breach, oversampled_side}``. On
        breach the lower-rate side is oversampled (p2 if ``p2_rate < p1_rate``).
        A perfectly balanced gap (e.g. 0) -> no breach -> 0.5/0.5,
        ``oversampled_side=None`` (no oversample change).
        """
        return second_start_oversampling_scheme(
            self.p1_score_rate(),
            self.p2_score_rate(),
            gap_threshold=self._gap_threshold,
        )

    # -- introspection ------------------------------------------------------
    def __len__(self) -> int:
        return len(self._results)

    def window_n(self) -> int:
        """The configured rolling-window capacity (deque maxlen)."""
        return int(self._results.maxlen)