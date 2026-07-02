"""Block E1 component E3 -- ``e1_tournament.py`` -- the tournament harness +
final-acceptance gate (E-E4..E-E9, E-E17, load-bearing).

V5-Max pipeline position: Block D COMPLETE (D1-D3 done) -> Block E1 (this file is
E3, the tournament + final-acceptance gate). E3 sits BETWEEN the Block-D exit
(D2 fills ``E1CandidateSet.post_d_path`` via ``with_post_d``) and the ship
component E5 (E5 ships the E3 winner). E3 picks the SINGLE ship winner among the
E1 candidate checkpoints (post-D first, post-C3 best, post-B fallback) by playing
a UNIFIED gauntlet (the A4 live runner, USER-provided at operational time) +
applying the FULL final-acceptance threshold table (SPEC :143-156).

WHY NEW (``BLOCK_E1_PLAN.md`` section E3): the final-acceptance gate is NOT
``run_v5_acceptance.py`` (a SCRIPT that plays NO games -- it reads pre-computed
winrates from a benchmark JSON + checks config FLAGS, ``run_v5_acceptance.py:488``
``candidate_no_assist_hidden_mode``). E3 builds the real tournament harness that
PLAYS the unified gauntlet (``a_gate.play_gauntlet``) + the side-stratified
gauntlet (``second_start_parity.play_side_stratified_gauntlet``) + composes the
A-gate / Block-B-gate criteria into the final threshold table. The two-notion
separation (ABSOLUTE vs V4-orig >= 0.70; PROGRESSION vs best self-snapshot ~0.52-0.55
monotone, SPEC :138-141) is LOAD-BEARING -- E3 keeps them as DISTINCT gate rows (a
candidate can pass progression but fail absolute, and vice versa).

COMPOSITION (A5/B5/B6 = oracle, E3 = UUT -- composes them, builds NEW
``E1TournamentConfig`` + ``E1CandidateReport``; does NOT mutate any composed
type):
  * REUSES A5 ``GateOutcome`` (``a_gate.py:151``) for each criterion row.
  * REUSES A5 ``ManaDrawBaseline`` (``a_gate.py:230``) for the mana_draw band.
  * REUSES A5 ``check_no_assist_gate`` / ``check_exploit_resistance_gate`` /
    ``check_mana_draw_band`` / ``check_h2h_trending`` / ``play_gauntlet`` /
    ``compute_score_rate`` (``a_gate.py``).
  * REUSES B5 ``play_side_stratified_gauntlet`` / ``SecondStartParityLoop``
    (``second_start_parity.py``) for the p1/p2 gap (optional side_runner).
  * REUSES B6 ``_check_h2h_vs_best`` (``block_b_gate.py:209``) -- the
    single-snapshot H2H rate check pattern -- READ-ONLY, NOT redefined. E3 calls
    it for the ABSOLUTE / sanity / progression-floor H2H rows (renaming the
    returned ``GateOutcome`` via ``dataclasses.replace``).
  * REUSES B6 ``_check_p1_p2_gap`` (``block_b_gate.py:257``) for the p1/p2 gap row.
  * CONSUMES ``c_to_d_handoff.E1CandidateSet`` (READ-ONLY) for the candidate paths.
  * The production ``GameRunner`` is USER-provided (``rust_live_self_play``); tests
    inject a fake duck-typed runner. E3 takes ``game_runner`` as a kwarg.

CONSTRAINTS (frozen-classic guard): NO edit to ``run_v5_acceptance.py`` /
``gauntlet_v5.py`` / ``league_v5.py`` / ``opponents_v5.py`` / ``a_gate.py`` /
``second_start_parity.py`` / ``block_b_*`` (all READ-ONLY -- E3 is a NEW sibling
COMPOSING them). ``league_v5.compare_adaptive_strength_monotonicity`` is NOT used
(synthetic proxy, ``a_gate.py:33-36``) -- a regression guard asserts it. The
winner selection consults ONLY external-bench criteria (the threshold-table
verdict; the human-QA soft verdict is a tiebreaker/advisory, NOT a hard gate per
E-E8). The tournament CAN fail (no-ship verdict -- ``select_e1_winner`` returns
None when no candidate passes).

Run: ``PYTHONPATH=.:TrainV3.5/python python3 -m pytest
TrainV3.5/python/train_v3/tests/test_e1_tournament.py``.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

# A5 = oracle. REUSE (read-only) the A5 pieces E3 composes:
# ``GateOutcome`` (the per-criterion result type), ``ManaDrawBaseline`` (Q4
# baseline B), ``GameRunner`` (the injectable A5 runner Protocol), the 4 named
# check_* gates, ``play_gauntlet`` (the gauntlet runner), ``compute_score_rate``,
# and the spec-raised threshold constants (NO_ASSIST / EXPLOIT_RESISTANCE /
# MANA_DRAW band).
from .a_gate import (
    EXPLOIT_AGENT_KINDS,
    EXPLOIT_RESISTANCE_MIN_SCORE_RATE,
    MANA_DRAW_BAND_HIGH,
    MANA_DRAW_BAND_LOW,
    NO_ASSIST_MIN_SCORE_RATE,
    GateOutcome,
    ManaDrawBaseline,
    check_exploit_resistance_gate,
    check_h2h_trending,
    check_mana_draw_band,
    check_no_assist_gate,
    compute_score_rate,
    play_gauntlet,
)
# B5 = oracle. REUSE (read-only) the side-stratified gauntlet + parity loop.
from .second_start_parity import (
    BLOCK_B_GAP_THRESHOLD,
    SecondStartParityLoop,
    play_side_stratified_gauntlet,
)
# B6 = oracle. REUSE (read-only) the single-snapshot H2H rate check pattern +
# the p1/p2 gap check. E3 does NOT redefine ``_check_h2h_vs_best`` (regression
# guard: it IMPORTS it from ``block_b_gate``).
from .block_b_gate import _check_h2h_vs_best, _check_p1_p2_gap
# D3 = oracle. CONSUME (read-only) the E1 candidate set.
from .c_to_d_handoff import E1CandidateSet

__all__ = [
    "H2H_V4_ORIG_MIN",
    "H2H_VS_RANDOM_MIN",
    "H2H_VS_END_TURN_MIN",
    "H2H_VS_SELF_SNAPSHOT_MIN_FLOOR",
    "P1_P2_MAX_SCORE_GAP",
    "MIN_E2E_TPS",
    "MIN_ENTROPY",
    "MAX_ABS_KL",
    "NO_BONUS_P1_MIN",
    "NO_BONUS_P2_MIN",
    "NO_BONUS_SECOND_MIN",
    "V4_MAX_ADVISORY_P1",
    "V4_MAX_ADVISORY_P2",
    "V4_MAX_ADVISORY_SECOND",
    "UNIFIED_GAUNTLET_ROSTER",
    "E1TournamentConfig",
    "E1CandidateReport",
    "run_e1_tournament",
    "select_e1_winner",
    "make_default_candidate_loader",
]


# =============================================================================
# Final-acceptance threshold constants (SPEC :143-156, BLOCK_E1_PLAN.md:111)
# =============================================================================
#: E-E4 -- the ABSOLUTE strength anchor: the candidate's H2H score rate vs the
#: frozen V4-orig must be >= this (user's original 70-80% band, retargeted to
#: V4-orig). NO constant exists today; E3 defines it.
H2H_V4_ORIG_MIN: float = 0.70

#: E-E7 -- the sanity gates: H2H vs Random / end_turn must be ~1.0; the gate
#: threshold is the spec floor (a candidate losing to Random/end_turn is
#: broken). NO constant exists today; E3 defines it.
H2H_VS_RANDOM_MIN: float = 0.95
H2H_VS_END_TURN_MIN: float = 0.95

#: E-E5 -- the PROGRESSION absolute floor (low end of the spec band 0.52-0.55,
#: ``design.md:138-141``): the candidate's LATEST H2H score rate vs the best
#: self-snapshot must be >= this (AND the series must be trending up). Distinct
#: from the ABSOLUTE anchor (two-notion separation -- a candidate can pass
#: progression but fail absolute, and vice versa).
H2H_VS_SELF_SNAPSHOT_MIN_FLOOR: float = 0.52

#: p1/p2 score-gap threshold (REUSE ``second_start_parity.BLOCK_B_GAP_THRESHOLD``
#: = ``PHASE_A_P1_P2_GAP_THRESHOLD`` = 0.12, ``ppo_phaseA_config.py:120``). The
#: <= direction mirrors SPEC :152 (consistent with how ``max_abs_kl<=0.12`` is
#: stated). E3 reuses the B5 constant so the gap definition is single-sourced.
P1_P2_MAX_SCORE_GAP: float = BLOCK_B_GAP_THRESHOLD

#: Throughput / entropy / KL floors (from ``run_v5_acceptance.py`` argparse
#: defaults :33/35/37; E3 defines them as module constants -- it does NOT import
#: the script). ``min_e2e_tps=12000``, ``min_entropy=0.70``, ``max_abs_kl=0.12``.
MIN_E2E_TPS: float = 12000.0
MIN_ENTROPY: float = 0.70
MAX_ABS_KL: float = 0.12

#: E-E6 -- the HARD no_bonus gates vs best self-snapshot (SPEC :149,
#: ``no_bonus p1/p2/second >= 0.70 each``). E-E6 retargets the no_bonus corridor
#: to the self-snapshot (the candidate must beat its best self-snapshot WITHOUT
#: its level handicap, from each start side). These are HARD -- a single side
#: below 0.70 fails the threshold table.
NO_BONUS_P1_MIN: float = 0.70
NO_BONUS_P2_MIN: float = 0.70
NO_BONUS_SECOND_MIN: float = 0.70

#: The SECONDARY advisory from the V4-max pre-baked JSON
#: (``run_v5_acceptance.py:41/43/45``). ``p1=0.75`` is STRICTER than SPEC :149's
#: 0.70-each -- a deliberate over-bench on the secondary read (NOT the spec
#: value). These are SOFT / advisory: they do NOT flip the aggregate verdict
#: (the HARD gate is the self-snapshot read above).
V4_MAX_ADVISORY_P1: float = 0.75
V4_MAX_ADVISORY_P2: float = 0.70
V4_MAX_ADVISORY_SECOND: float = 0.70

#: E-E17 -- the UNIFIED gauntlet roster: the candidate plays vs V4-orig (the
#: ABSOLUTE anchor), Random + end_turn (the sanity gates), best_self_snapshot
#: (the PROGRESSION + no_bonus HARD gate reference), and the 7 EXPLOIT_AGENT_KINDS
#: (the no_assist / exploit_resistance gates). ``best_self_snapshot`` is the E3
#: name for the best-self-snapshot lane (Block B's ``best_ever`` reference,
#: ``block_d_league_driver._DEFAULT_H2H_OPPONENT_KIND`` -- E3 does NOT retarget
#: it, it reuses the Block-B naming).
UNIFIED_GAUNTLET_ROSTER: tuple[str, ...] = (
    "v4max",
    "random",
    "end_turn",
    "best_self_snapshot",
    *EXPLOIT_AGENT_KINDS,
)


# =============================================================================
# Config + report dataclasses
# =============================================================================
@dataclass(frozen=True)
class E1TournamentConfig:
    """The tournament configuration (``BLOCK_E1_PLAN.md:112``).

    ``candidate_set`` is the ``E1CandidateSet`` (D3 + D2) consumed READ-ONLY --
    E3 iterates the candidate paths in order (post-D first, then post-C3, then
    post-B; Nones dropped; dedup). ``mana_draw_baseline`` is the Q4 human
    baseline B (``a_gate.ManaDrawBaseline``) -- REQUIRED (the mana_draw-band
    gate is undefined without B; production supplies the real measured B).
    ``gauntlet_roster`` is E-E17 (the unified roster; default
    ``UNIFIED_GAUNTLET_ROSTER``). ``games_per_opponent`` is the A5
    ``play_gauntlet`` game count per opponent; ``games_per_opponent_per_side``
    is the B5 ``play_side_stratified_gauntlet`` game count per opponent per
    candidate side (p1/p2). ``seeds`` is the per-snapshot seed tuple (the first
    seed drives the unified gauntlet; the candidate's prior H2H-vs-self-snapshot
    history is threaded via the candidate metadata, NOT re-played). The
    throughput/entropy/KL floors are the module constants by default. The
    ``no_bonus_benchmark_json_path`` is the V4-max pre-baked JSON (E-E6
    SECONDARY advisory) -- None when no advisory JSON is supplied.
    """

    candidate_set: E1CandidateSet
    mana_draw_baseline: ManaDrawBaseline
    games_per_opponent: int = 20
    games_per_opponent_per_side: int = 10
    seeds: tuple[int, ...] = (0,)
    gauntlet_roster: tuple[str, ...] = UNIFIED_GAUNTLET_ROSTER
    throughput_floor: float = MIN_E2E_TPS
    entropy_floor: float = MIN_ENTROPY
    max_abs_kl_floor: float = MAX_ABS_KL
    no_bonus_benchmark_json_path: Optional[str] = None


@dataclass(frozen=True)
class E1CandidateReport:
    """The per-candidate tournament report (``BLOCK_E1_PLAN.md:113``).

    Carries the measured H2H vs each lane (V4-orig / Random / end_turn /
    best-self-snapshot latest + the trending series), the no_assist /
    exploit_resistance score rates, the mana_draw rate + baseline, the p1/p2
    gap, the no_bonus p1/p2/second (HARD self-snapshot read + the V4-max
    SECONDARY advisory, Optional), the throughput/entropy/KL, the human-QA
    verdict (E4, SOFT -- NOT a hard gate), the per-criterion ``GateOutcome`` map
    (the SPEC :143-156 threshold table), and the aggregate verdict
    (``"pass"`` / ``"fail"`` per candidate; a NO-SHIP verdict is the
    tournament-level outcome -- ``select_e1_winner`` returns ``None`` when no
    candidate passes, NOT a per-candidate ``aggregate_verdict`` value).
    """

    candidate_path: str
    h2h_vs_v4_orig: float
    h2h_vs_random: float
    h2h_vs_end_turn: float
    h2h_vs_self_snapshot_latest: float
    h2h_vs_self_snapshot_trending: list[float]
    no_assist_score_rate: float
    exploit_resistance_score_rate: float
    mana_draw_rate: float
    mana_draw_baseline: ManaDrawBaseline
    p1_p2_gap: float
    no_bonus_p1: float
    no_bonus_p2: float
    no_bonus_second: float
    no_bonus_advisory_p1: Optional[float]
    no_bonus_advisory_p2: Optional[float]
    no_bonus_advisory_second: Optional[float]
    throughput: float
    entropy: float
    max_abs_kl: float
    human_qa_verdict: Any
    per_criterion: dict[str, GateOutcome]
    aggregate_verdict: str

    def passed(self) -> bool:
        """True iff the aggregate verdict is ``"pass"`` (all HARD gates passed)."""
        return self.aggregate_verdict == "pass"

    def failed_criteria(self) -> list[str]:
        """Names of the HARD criterion rows that failed (empty iff ``passed``).

        Filters to ``_HARD_ROWS`` only -- the SOFT advisory rows
        (``no_bonus_advisory_*``) + the human-QA verdict are NOT HARD and must
        NOT appear here (a candidate that passes every HARD gate but fails a
        SOFT advisory still has ``passed() == True`` + an empty
        ``failed_criteria()``)."""
        return [
            name for name, o in self.per_criterion.items()
            if name in _HARD_ROWS and not o.passed
        ]


# =============================================================================
# Candidate loader (E-E18 = model_mlx.load_checkpoint; production-wired)
# =============================================================================
class _CandidateLoader(Protocol):
    def __call__(self, path: str) -> dict: ...


def make_default_candidate_loader(policy: Any) -> Callable[[str], dict]:
    """Build the production candidate loader (E-E18 = ``model_mlx.load_checkpoint``,
    V5-native). Returns a callable ``(path) -> dict`` that loads the checkpoint
    metadata for ``path`` into the supplied V5 ``policy``. Production calls this
    with the real V5 ``V5ActionConditionedPolicy``; tests inject their own fake
    loader (so E3 never needs MLX/Rust/ONNX to unit-test the gate logic).

    The loader returns ``{"metadata": {...}}`` (``model_mlx.load_checkpoint``
    return shape, ``ai/train_v2/model_mlx.py:117``). E3 reads the candidate's
    run-artifact fields (throughput / entropy / max_abs_kl / no_bonus p1/p2/
    second / the prior H2H-vs-self-snapshot history / p1_p2_gap) from the
    metadata dict.
    """
    from ai.train_v2.model_mlx import load_checkpoint  # READ-ONLY (E-E18)

    def _load(path: str) -> dict:
        return load_checkpoint(path, policy)

    return _load


# =============================================================================
# Helpers
# =============================================================================
def _candidate_paths(candidate_set: E1CandidateSet) -> list[str]:
    """Iterate the E1 candidate paths in order (post-D first, then post-C3, then
    post-B; Nones dropped; dedup preserving order) -- ``BLOCK_E1_PLAN.md:114``."""
    raw = [
        candidate_set.post_d_path,
        candidate_set.post_c3_best_path,
        candidate_set.post_b_path,
    ]
    seen: set[str] = set()
    out: list[str] = []
    for p in raw:
        if p is None:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _opp_score_rate(per_opponent: dict[str, dict[str, int]], opp: str) -> float:
    """Score rate for a single opponent lane from ``GauntletOutcomes.per_opponent``."""
    stats = per_opponent[opp]
    return compute_score_rate(stats["wins"], stats["draws"], stats["losses"])


def _roster_score_rate(
    per_opponent: dict[str, dict[str, int]],
    kinds: tuple[str, ...],
) -> float:
    """Aggregate score rate over a subset of the roster (sum wins/draws/losses
    across the listed lanes, then ``compute_score_rate``)."""
    w = d = l = 0
    for k in kinds:
        stats = per_opponent[k]
        w += int(stats["wins"])
        d += int(stats["draws"])
        l += int(stats["losses"])
    return compute_score_rate(w, d, l)


def _rename(outcome: GateOutcome, *, name: str, details: dict) -> GateOutcome:
    """Return a NEW ``GateOutcome`` with a renamed criterion + merged details.

    REUSES the A5 ``GateOutcome`` (frozen dataclass) via ``dataclasses.replace``
    -- E3 does NOT redefine ``GateOutcome`` (regression guard). Used to adapt the
    B6 ``_check_h2h_vs_best`` result (name ``"h2h_vs_best"``) to the E3-specific
    criterion names (``h2h_vs_v4_orig`` / ``h2h_vs_random`` / ...)."""
    merged = {**outcome.details, **details}
    return dataclasses.replace(outcome, name=name, details=merged)


def _check_floor(value: float, *, threshold: float, name: str) -> GateOutcome:
    """Inline ``>= threshold`` floor check for the throughput/entropy/no_bonus
    rows (the A5/B6 helpers do not cover these E3-specific metrics). REUSES
    ``GateOutcome`` (does NOT redefine it)."""
    value = float(value)
    passed = value >= float(threshold)
    return GateOutcome(
        name=name,
        passed=passed,
        score=value,
        threshold=float(threshold),
        details={"criterion": name, "direction": ">="},
    )


def _check_ceiling(value: float, *, threshold: float, name: str) -> GateOutcome:
    """Inline ``<= threshold`` ceiling check for the max_abs_kl row. REUSES
    ``GateOutcome`` (does NOT redefine it)."""
    value = float(value)
    passed = value <= float(threshold)
    return GateOutcome(
        name=name,
        passed=passed,
        score=value,
        threshold=float(threshold),
        details={"criterion": name, "direction": "<="},
    )


def _load_advisory(
    json_path: Optional[str],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Read the V4-max pre-baked no_bonus advisory (p1/p2/second) from the JSON
    path. Returns ``(None, None, None)`` when ``json_path`` is None or the file
    is missing / malformed (the advisory is SOFT -- a missing JSON does NOT flip
    the verdict)."""
    if json_path is None:
        return None, None, None
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None, None, None
    return (
        data.get("no_bonus_p1"),
        data.get("no_bonus_p2"),
        data.get("no_bonus_second"),
    )


# The HARD criterion row names (the rows that flip the aggregate verdict). The
# advisory rows (``no_bonus_advisory_*``) + the human-QA verdict are SOFT.
_HARD_ROWS: tuple[str, ...] = (
    "h2h_vs_v4_orig",
    "h2h_vs_random",
    "h2h_vs_end_turn",
    "h2h_vs_self_snapshot_trending",
    "h2h_vs_self_snapshot_floor",
    "no_assist",
    "exploit_resistance",
    "mana_draw_band",
    "p1_p2_gap",
    "throughput",
    "entropy",
    "max_abs_kl",
    "no_bonus_p1",
    "no_bonus_p2",
    "no_bonus_second",
)


# =============================================================================
# run_e1_tournament -- the tournament harness + final-acceptance gate
# =============================================================================
def run_e1_tournament(
    config: E1TournamentConfig,
    *,
    game_runner: Any,
    candidate_loader: Callable[[str], dict],
    side_runner: Any = None,
) -> list[E1CandidateReport]:
    """Run the E1 tournament over the candidate set + emit the per-candidate
    final-acceptance threshold table (``BLOCK_E1_PLAN.md:114``).

    For each candidate path in ``config.candidate_set`` (Nones dropped, post-D
    first, dedup):
      1. load the candidate via ``candidate_loader`` (E-E18 =
         ``model_mlx.load_checkpoint``, V5-native; tests inject a fake). The
         returned ``{"metadata": {...}}`` carries the candidate's run-artifact
         fields (throughput / entropy / max_abs_kl / no_bonus p1/p2/second / the
         prior H2H-vs-self-snapshot history / p1_p2_gap when no side_runner).
      2. play the unified gauntlet via ``a_gate.play_gauntlet`` with
         ``config.gauntlet_roster`` + ``config.games_per_opponent``; extract the
         per-lane score rates (v4max -> ``h2h_vs_v4_orig``, random ->
         ``h2h_vs_random``, end_turn -> ``h2h_vs_end_turn``, best_self_snapshot
         -> ``h2h_vs_self_snapshot_latest``) + the no_assist / exploit_resistance
         rates (over ``EXPLOIT_AGENT_KINDS``) + the mana_draw rate.
      3. build the H2H-vs-self-snapshot trending series = the candidate's prior
         history (metadata) + the just-measured latest rate; apply
         ``a_gate.check_h2h_trending`` + the NEW
         ``H2H_VS_SELF_SNAPSHOT_MIN_FLOOR=0.52`` absolute-floor check (E-E5 =
         trending AND latest >= 0.52 -- two DISTINCT rows for the two-notion
         separation).
      4. apply the A-gate no_assist / exploit_resistance / mana_draw-band gates
         (REUSED from ``a_gate``), the V4-orig / Random / end_turn H2H gates
         (E-E4 0.70 / E-E7 0.95 -- via the B6 ``_check_h2h_vs_best`` reuse), the
         p1/p2 gap gate (<= 0.12, via the B6 ``_check_p1_p2_gap`` reuse), the
         throughput / entropy / KL floors, the no_bonus HARD gates (>= 0.70 each,
         E-E6) + the V4-max SECONDARY advisory.
      5. emit the SPEC :143-156 threshold table per candidate (the
         ``per_criterion`` ``GateOutcome`` map + the aggregate verdict).

    The side-stratified p1/p2 gap is MEASURED via
    ``second_start_parity.play_side_stratified_gauntlet`` when ``side_runner`` is
    supplied (production wires an A4 live-runner adapter); when ``side_runner``
    is None the p1/p2 gap is read from the candidate metadata (the training run
    measured it). The no_bonus p1/p2/second (the HARD self-snapshot read) are
    read from the candidate metadata in both cases (they are a SEPARATE
    no-handicap benchmark, NOT the parity gauntlet).

    ``game_runner`` is the A5 ``GameRunner`` (``play(opponent_kind, *, seed)``);
    ``side_runner`` is the B5 ``BlockBGameRunner``
    (``play(opponent_kind, *, seed, candidate_side)``). Tests inject fakes;
    production wires the A4 ``rust_live_self_play`` runner (USER-provided).
    """
    candidate_paths = _candidate_paths(config.candidate_set)
    if not candidate_paths:
        return []

    seed = int(config.seeds[0]) if config.seeds else 0
    baseline = config.mana_draw_baseline
    if baseline is None:
        raise ValueError(
            "E1TournamentConfig.mana_draw_baseline is required (the Q4 human "
            "baseline B must be supplied -- the mana_draw-band gate is "
            "undefined without it)"
        )
    advisory_p1, advisory_p2, advisory_second = _load_advisory(
        config.no_bonus_benchmark_json_path
    )

    reports: list[E1CandidateReport] = []
    for path in candidate_paths:
        loaded = candidate_loader(path)
        meta = (loaded or {}).get("metadata", {}) if isinstance(loaded, dict) else {}

        # --- play the unified gauntlet (A5 play_gauntlet) ----------------------
        outcomes = play_gauntlet(
            game_runner,
            list(config.gauntlet_roster),
            games_per_opponent=int(config.games_per_opponent),
            seed=seed,
        )
        per_opp = outcomes.per_opponent
        h2h_v4_orig = _opp_score_rate(per_opp, "v4max")
        h2h_random = _opp_score_rate(per_opp, "random")
        h2h_end_turn = _opp_score_rate(per_opp, "end_turn")
        h2h_self_latest = _opp_score_rate(per_opp, "best_self_snapshot")
        # no_assist + exploit_resistance are INDEPENDENT measurements (separate
        # gauntlets / modes, harvested by the training run) -- read from metadata
        # so the two rates are independent (the unified gauntlet's exploit-kinds
        # lanes are NOT a no-assist-mode measurement). Fallback to the gauntlet
        # aggregate when metadata does not carry them.
        no_assist_rate = float(
            meta.get("no_assist_score_rate", outcomes.score_rate())
        )
        exploit_rate = float(
            meta.get(
                "exploit_resistance_score_rate",
                _roster_score_rate(per_opp, tuple(EXPLOIT_AGENT_KINDS)),
            )
        )
        mana_draw_rate = outcomes.mana_draw_rate()

        # --- the progression trending series (prior history + latest) ----------
        prior_history = [float(x) for x in meta.get("h2h_vs_self_snapshot_history", [])]
        trending_series = prior_history + [h2h_self_latest]

        # --- the p1/p2 gap (measured via side_runner OR read from metadata) ----
        if side_runner is not None:
            side_results = play_side_stratified_gauntlet(
                side_runner,
                ["best_self_snapshot"],
                games_per_opponent_per_side=int(config.games_per_opponent_per_side),
                seed=seed,
            )
            loop = SecondStartParityLoop(
                window_n=max(int(config.games_per_opponent_per_side) * 2, 1)
            )
            loop.update(side_results)
            p1_p2_gap = float(loop.gap())
        else:
            p1_p2_gap = float(meta.get("p1_p2_gap", 0.0))

        # --- run-artifact fields (throughput / entropy / KL / no_bonus) -------
        throughput = float(meta.get("throughput", 0.0))
        entropy = float(meta.get("entropy", 0.0))
        max_abs_kl = float(meta.get("max_abs_kl", 0.0))
        no_bonus_p1 = float(meta.get("no_bonus_p1", 0.0))
        no_bonus_p2 = float(meta.get("no_bonus_p2", 0.0))
        no_bonus_second = float(meta.get("no_bonus_second", 0.0))
        human_qa = meta.get("human_qa_verdict")

        # --- build the per-criterion threshold table (SPEC :143-156) ----------
        per_criterion: dict[str, GateOutcome] = {}

        # E-E4 ABSOLUTE anchor (V4-orig >= 0.70) -- DISTINCT from progression.
        per_criterion["h2h_vs_v4_orig"] = _rename(
            _check_h2h_vs_best(h2h_v4_orig, threshold=H2H_V4_ORIG_MIN),
            name="h2h_vs_v4_orig",
            details={"criterion": "h2h_vs_v4_orig_absolute_anchor", "notion": "absolute"},
        )
        # E-E7 sanity gates (Random / end_turn ~1.0).
        per_criterion["h2h_vs_random"] = _rename(
            _check_h2h_vs_best(h2h_random, threshold=H2H_VS_RANDOM_MIN),
            name="h2h_vs_random",
            details={"criterion": "h2h_vs_random_sanity", "notion": "sanity"},
        )
        per_criterion["h2h_vs_end_turn"] = _rename(
            _check_h2h_vs_best(h2h_end_turn, threshold=H2H_VS_END_TURN_MIN),
            name="h2h_vs_end_turn",
            details={"criterion": "h2h_vs_end_turn_sanity", "notion": "sanity"},
        )
        # E-E5 PROGRESSION -- trending (check_h2h_trending) + the absolute floor
        # (latest >= 0.52) as TWO DISTINCT rows (the two-notion separation is
        # ABSOLUTE-vs-PROGRESSION; the floor is the low end of the 0.52-0.55 band).
        per_criterion["h2h_vs_self_snapshot_trending"] = check_h2h_trending(
            trending_series
        )
        per_criterion["h2h_vs_self_snapshot_floor"] = _rename(
            _check_h2h_vs_best(
                h2h_self_latest, threshold=H2H_VS_SELF_SNAPSHOT_MIN_FLOOR
            ),
            name="h2h_vs_self_snapshot_floor",
            details={
                "criterion": "h2h_vs_self_snapshot_floor",
                "notion": "progression_floor",
            },
        )
        # A-gate no_assist / exploit_resistance / mana_draw_band (REUSED).
        per_criterion["no_assist"] = check_no_assist_gate(no_assist_rate)
        per_criterion["exploit_resistance"] = check_exploit_resistance_gate(
            exploit_rate
        )
        per_criterion["mana_draw_band"] = check_mana_draw_band(
            mana_draw_rate, baseline
        )
        # p1/p2 gap (B6 _check_p1_p2_gap reuse -- <= 0.12).
        per_criterion["p1_p2_gap"] = _check_p1_p2_gap(
            p1_p2_gap, threshold=P1_P2_MAX_SCORE_GAP
        )
        # Throughput / entropy / KL floors.
        per_criterion["throughput"] = _check_floor(
            throughput, threshold=config.throughput_floor, name="throughput"
        )
        per_criterion["entropy"] = _check_floor(
            entropy, threshold=config.entropy_floor, name="entropy"
        )
        per_criterion["max_abs_kl"] = _check_ceiling(
            max_abs_kl, threshold=config.max_abs_kl_floor, name="max_abs_kl"
        )
        # E-E6 no_bonus HARD gates (self-snapshot, >= 0.70 each).
        per_criterion["no_bonus_p1"] = _check_floor(
            no_bonus_p1, threshold=NO_BONUS_P1_MIN, name="no_bonus_p1"
        )
        per_criterion["no_bonus_p2"] = _check_floor(
            no_bonus_p2, threshold=NO_BONUS_P2_MIN, name="no_bonus_p2"
        )
        per_criterion["no_bonus_second"] = _check_floor(
            no_bonus_second, threshold=NO_BONUS_SECOND_MIN, name="no_bonus_second"
        )
        # SOFT advisory rows (V4-max pre-baked) -- recorded but NOT in _HARD_ROWS.
        if advisory_p1 is not None:
            per_criterion["no_bonus_advisory_p1"] = _check_floor(
                float(advisory_p1), threshold=V4_MAX_ADVISORY_P1,
                name="no_bonus_advisory_p1",
            )
        if advisory_p2 is not None:
            per_criterion["no_bonus_advisory_p2"] = _check_floor(
                float(advisory_p2), threshold=V4_MAX_ADVISORY_P2,
                name="no_bonus_advisory_p2",
            )
        if advisory_second is not None:
            per_criterion["no_bonus_advisory_second"] = _check_floor(
                float(advisory_second), threshold=V4_MAX_ADVISORY_SECOND,
                name="no_bonus_advisory_second",
            )

        # --- aggregate verdict (ONLY the HARD rows flip it) -------------------
        all_hard_pass = all(
            per_criterion[name].passed for name in _HARD_ROWS if name in per_criterion
        )
        aggregate_verdict = "pass" if all_hard_pass else "fail"

        reports.append(
            E1CandidateReport(
                candidate_path=path,
                h2h_vs_v4_orig=h2h_v4_orig,
                h2h_vs_random=h2h_random,
                h2h_vs_end_turn=h2h_end_turn,
                h2h_vs_self_snapshot_latest=h2h_self_latest,
                h2h_vs_self_snapshot_trending=trending_series,
                no_assist_score_rate=no_assist_rate,
                exploit_resistance_score_rate=exploit_rate,
                mana_draw_rate=mana_draw_rate,
                mana_draw_baseline=baseline,
                p1_p2_gap=p1_p2_gap,
                no_bonus_p1=no_bonus_p1,
                no_bonus_p2=no_bonus_p2,
                no_bonus_second=no_bonus_second,
                no_bonus_advisory_p1=advisory_p1,
                no_bonus_advisory_p2=advisory_p2,
                no_bonus_advisory_second=advisory_second,
                throughput=throughput,
                entropy=entropy,
                max_abs_kl=max_abs_kl,
                human_qa_verdict=human_qa,
                per_criterion=per_criterion,
                aggregate_verdict=aggregate_verdict,
            )
        )

    return reports


# =============================================================================
# select_e1_winner -- the promotion selector (promotion-by-loss guard inherited)
# =============================================================================
def select_e1_winner(
    reports: list[E1CandidateReport],
) -> Optional[E1CandidateReport]:
    """The E1 winner selection (``BLOCK_E1_PLAN.md:115``).

    The winner is the candidate that PASSES the FULL threshold table
    (``aggregate_verdict == "pass"``); among multiple passers, the HIGHEST
    ``h2h_vs_v4_orig`` (the ABSOLUTE anchor) wins. If NONE pass, returns None --
    the NO-SHIP verdict (the tournament CAN fail; this is the final gate).

    The selection consults ONLY external-bench criteria (the threshold-table
    verdict; ``h2h_vs_v4_orig`` is the tiebreaker among passers). The human-QA
    soft verdict is a tiebreaker/advisory, NOT a hard gate (E-E8): a candidate
    that passes the threshold table but has a ``human_qa_verdict.verdict ==
    "easier"`` STILL PASSES (the hard ship decision is the threshold table).

    Promotion-by-loss guard (inherited from A5 ``select_promotion``,
    ``a_gate.py:638-641``): internal training metrics (PPO loss / KL / entropy)
    are MONITORING-ONLY and are NOT consulted -- the decision never reads them.
    """
    passers = [r for r in reports if r.passed()]
    if not passers:
        return None
    # Highest ABSOLUTE anchor (h2h_vs_v4_orig) among passers wins. Ties break by
    # candidate_path (stable, deterministic) -- the human-QA soft verdict is NOT
    # a hard gate (E-E8) so it is NOT consulted here.
    return max(passers, key=lambda r: (r.h2h_vs_v4_orig, r.candidate_path))