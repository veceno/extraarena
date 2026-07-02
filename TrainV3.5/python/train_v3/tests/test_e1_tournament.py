"""Synthetic tests for Block E1 component E3 -- ``e1_tournament.py`` (the
tournament harness + final-acceptance gate). TRACKED.

All tests use SYNTHETIC game outcomes via a fake ``GameRunner`` returning canned
``GameResult``s + a fake ``candidate_loader`` returning canned metadata -- the
threshold-table verdict logic is unit-testable WITHOUT MLX/Rust/ONNX (no real
Rust arena, no real policy, no real V5 checkpoint). This mirrors the A5
fake-runner pattern (``test_a_gate.py:59-86``) + the B5 fake-runner pattern
(``test_second_start_parity.py:36-65``).

The fake ``GameRunner`` uses an explicit per-opponent-outcome-sequence model so
the per-lane score rates (v4max / random / end_turn / best_self_snapshot) are
DETERMINISTIC and the threshold-table verdict is asserted EXACTLY. The
no_assist / exploit_resistance / p1_p2_gap / no_bonus / throughput / entropy /
max_abs_kl / prior H2H-vs-self-snapshot history are read from the fake candidate
metadata (independent measurements harvested by the training run).

Run: ``PYTHONPATH="/path/to/worktree:/path/to/worktree/TrainV3.5/python" python3
-m pytest TrainV3.5/python/train_v3/tests/test_e1_tournament.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# sys.path bootstrap: insert the worktree root (so ``ai.train_v2.model_mlx``
# resolves) AND the TrainV3.5/python parent (so ``train_v3.*`` resolves) when run
# via ``python -m pytest`` from the worktree root. Mirrors the Block D test
# pattern (``test_c_to_d_handoff.py:29-31`` inserts the train_v3 parent; E3 ALSO
# needs ``ai.train_v2.model_mlx`` so insert the worktree root too).
_HERE = Path(__file__).resolve()
_TV3_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# The worktree root is 4 levels up from the test file:
# .../glm-TrainV3.5Prep/TrainV3.5/python/train_v3/tests/test_e1_tournament.py
_WORKTREE_ROOT = os.path.abspath(str(_HERE.parents[4]))
for _p in (_TV3_PARENT, _WORKTREE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_v3.a_gate import (  # noqa: E402
    EXPLOIT_AGENT_KINDS,
    GameResult,
    ManaDrawBaseline,
    record_mana_draw_baseline,
)
from train_v3.c_to_d_handoff import E1CandidateSet  # noqa: E402
from train_v3.e1_tournament import (  # noqa: E402
    H2H_V4_ORIG_MIN,
    H2H_VS_END_TURN_MIN,
    H2H_VS_RANDOM_MIN,
    H2H_VS_SELF_SNAPSHOT_MIN_FLOOR,
    MAX_ABS_KL,
    MIN_E2E_TPS,
    MIN_ENTROPY,
    NO_BONUS_P1_MIN,
    NO_BONUS_P2_MIN,
    NO_BONUS_SECOND_MIN,
    P1_P2_MAX_SCORE_GAP,
    UNIFIED_GAUNTLET_ROSTER,
    E1CandidateReport,
    E1TournamentConfig,
    run_e1_tournament,
    select_e1_winner,
)


# =============================================================================
# Fakes
# =============================================================================
def _outcomes(rate: float, n: int = 100) -> list[str]:
    """Build a length-``n`` outcome list (wins + losses, NO draws) with score
    rate == ``rate`` (``rate*n`` wins). ``rate*n`` must be integer-rounded; for
    the test rates (0.72, 0.97, ...) with n=100 this is exact."""
    w = round(float(rate) * n)
    assert abs(w / n - float(rate)) < 1e-9, f"rate {rate} not expressible at n={n}"
    return ["win"] * w + ["loss"] * (n - w)


class _FakeGameRunner:
    """Deterministic fake ``GameRunner``: returns the next outcome from
    ``per_opponent_outcomes[opponent_kind]`` (cycling modulo so multi-candidate
    tournaments never exhaust the list and each candidate sees the SAME rate).
    ``mana_draw_count`` / ``eligible_turns`` are constant per game so the
    aggregate mana_draw rate is exactly ``mana_draw_count / eligible_turns``."""

    def __init__(
        self,
        per_opponent_outcomes: dict[str, list[str]] | None = None,
        *,
        mana_draw_count: int = 4,
        eligible_turns: int = 10,
    ) -> None:
        self.per_opponent_outcomes = per_opponent_outcomes or {}
        self.mana_draw_count = int(mana_draw_count)
        self.eligible_turns = int(eligible_turns)
        self._idx: dict[str, int] = {}
        self.calls: list[tuple[str, int]] = []

    def play(self, opponent_kind: str, *, seed: int) -> GameResult:
        self.calls.append((opponent_kind, seed))
        lst = self.per_opponent_outcomes.get(opponent_kind, ["draw"])
        i = self._idx.get(opponent_kind, 0)
        outcome = lst[i % len(lst)]
        self._idx[opponent_kind] = i + 1
        return GameResult(
            outcome=outcome,
            mana_draw_count=self.mana_draw_count,
            eligible_turns=self.eligible_turns,
            opponent=opponent_kind,
        )


class _FakeCandidateLoader:
    """Fake ``candidate_loader``: records the paths called + returns canned
    metadata. Mirrors the ``model_mlx.load_checkpoint`` return shape
    ``{"metadata": {...}}`` (``ai/train_v2/model_mlx.py:117``)."""

    def __init__(self, meta: dict) -> None:
        self.meta = dict(meta)
        self.calls: list[str] = []

    def __call__(self, path: str) -> dict:
        self.calls.append(path)
        return {"metadata": dict(self.meta)}


class _HumanQAFake:
    """A minimal stand-in for E4's ``HumanQAVerdict`` (E4 not yet written); E3
    types ``human_qa_verdict`` as ``Any`` so any object with a ``.verdict``
    attribute works."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict


# =============================================================================
# Builders
# =============================================================================
def _per_opp(
    *,
    v4max: float = 0.72,
    random: float = 0.97,
    end_turn: float = 0.96,
    best_self: float = 0.54,
    n: int = 100,
) -> dict[str, list[str]]:
    """Build the per-opponent outcome map for the 4 anchor lanes; the 7 exploit
    kinds default to all-draws (their outcomes do NOT affect the verdict --
    exploit_resistance is metadata-sourced; they only feed the mana_draw
    aggregate which is constant per game)."""
    return {
        "v4max": _outcomes(v4max, n),
        "random": _outcomes(random, n),
        "end_turn": _outcomes(end_turn, n),
        "best_self_snapshot": _outcomes(best_self, n),
        **{k: ["draw"] * n for k in EXPLOIT_AGENT_KINDS},
    }


def _meta(
    *,
    history=(0.50, 0.51, 0.52, 0.53),
    no_assist=0.56,
    exploit=0.51,
    p1_p2_gap=0.10,
    throughput=12500.0,
    entropy=0.72,
    max_abs_kl=0.11,
    no_bonus_p1=0.76,
    no_bonus_p2=0.71,
    no_bonus_second=0.71,
    human_qa=None,
) -> dict:
    return {
        "h2h_vs_self_snapshot_history": list(history),
        "no_assist_score_rate": float(no_assist),
        "exploit_resistance_score_rate": float(exploit),
        "p1_p2_gap": float(p1_p2_gap),
        "throughput": float(throughput),
        "entropy": float(entropy),
        "max_abs_kl": float(max_abs_kl),
        "no_bonus_p1": float(no_bonus_p1),
        "no_bonus_p2": float(no_bonus_p2),
        "no_bonus_second": float(no_bonus_second),
        "human_qa_verdict": human_qa,
    }


def _baseline() -> ManaDrawBaseline:
    """Q4 baseline B = 0.4 -> band [0.2, 0.6]; the fake runner's 4/10 = 0.4 is in
    band."""
    return record_mana_draw_baseline(40, 100)


def _run_one(
    per_opp: dict[str, list[str]] | None = None,
    meta: dict | None = None,
    *,
    path: str = "cand_post_d",
    config: E1TournamentConfig | None = None,
) -> E1CandidateReport:
    """Run a single-candidate tournament + return its report."""
    if per_opp is None:
        per_opp = _per_opp()
    if meta is None:
        meta = _meta()
    runner = _FakeGameRunner(per_opponent_outcomes=per_opp)
    loader = _FakeCandidateLoader(meta)
    if config is None:
        config = E1TournamentConfig(
            candidate_set=E1CandidateSet(post_d_path=path),
            mana_draw_baseline=_baseline(),
            games_per_opponent=100,
            games_per_opponent_per_side=10,
            seeds=(0,),
        )
    reports = run_e1_tournament(
        config, game_runner=runner, candidate_loader=loader
    )
    assert len(reports) == 1, f"expected 1 report, got {len(reports)}"
    return reports[0]


# =============================================================================
# Tests
# =============================================================================
def test_passing_candidate() -> None:
    """The canonical passing candidate (SPEC :143-156): v4max 0.72, random 0.97,
    end_turn 0.96, best-self trending [0.50..0.54] latest 0.54, no_assist 0.56,
    exploit 0.51, mana_draw in band, gap 0.10, throughput 12500, entropy 0.72,
    kl 0.11, no_bonus 0.76/0.71/0.71 -> aggregate 'pass'."""
    report = _run_one()
    assert report.aggregate_verdict == "pass", report.failed_criteria()
    assert report.h2h_vs_v4_orig == pytest.approx(0.72)
    assert report.h2h_vs_random == pytest.approx(0.97)
    assert report.h2h_vs_end_turn == pytest.approx(0.96)
    assert report.h2h_vs_self_snapshot_latest == pytest.approx(0.54)
    assert report.h2h_vs_self_snapshot_trending == [0.50, 0.51, 0.52, 0.53, 0.54]
    assert report.passed()


def test_fail_absolute_anchor() -> None:
    """v4max 0.68 (<0.70) -> aggregate 'fail' (the ABSOLUTE anchor fails even if
    progression passes)."""
    report = _run_one(
        _per_opp(v4max=0.68, best_self=0.54),
        _meta(history=(0.50, 0.51, 0.52, 0.53)),
    )
    assert report.aggregate_verdict == "fail"
    assert not report.per_criterion["h2h_vs_v4_orig"].passed
    # progression rows pass (two-notion separation: absolute fails, progression passes)
    assert report.per_criterion["h2h_vs_self_snapshot_floor"].passed
    assert report.per_criterion["h2h_vs_self_snapshot_trending"].passed
    assert "h2h_vs_v4_orig" in report.failed_criteria()


def test_fail_progression_floor() -> None:
    """best_self latest 0.51 (<0.52) -> aggregate 'fail' (progression floor fails
    even if absolute passes). Trending is kept PASSING so the FLOOR is the
    isolated cause (two-notion separation: progression fails, absolute passes)."""
    report = _run_one(
        _per_opp(v4max=0.72, best_self=0.51),
        _meta(history=(0.48, 0.49, 0.50, 0.51)),
    )
    assert report.aggregate_verdict == "fail"
    assert report.per_criterion["h2h_vs_v4_orig"].passed  # absolute passes
    assert report.per_criterion["h2h_vs_self_snapshot_trending"].passed  # trending passes
    assert not report.per_criterion["h2h_vs_self_snapshot_floor"].passed  # floor fails
    assert "h2h_vs_self_snapshot_floor" in report.failed_criteria()


def test_fail_sanity() -> None:
    """random 0.94 (<0.95) -> aggregate 'fail' (sanity gate fails)."""
    report = _run_one(_per_opp(random=0.94), _meta())
    assert report.aggregate_verdict == "fail"
    assert not report.per_criterion["h2h_vs_random"].passed
    assert "h2h_vs_random" in report.failed_criteria()


def test_two_notion_separation() -> None:
    """BOTH directions of the two-notion separation are load-bearing:
    (A) passes progression but fails absolute = FAIL;
    (B) passes absolute but fails progression = FAIL."""
    # (A) progression passes, absolute fails.
    a = _run_one(
        _per_opp(v4max=0.68, best_self=0.54),
        _meta(history=(0.50, 0.51, 0.52, 0.53)),
        path="cand_a",
    )
    assert a.aggregate_verdict == "fail"
    assert a.per_criterion["h2h_vs_self_snapshot_floor"].passed
    assert not a.per_criterion["h2h_vs_v4_orig"].passed
    # (B) absolute passes, progression fails.
    b = _run_one(
        _per_opp(v4max=0.72, best_self=0.51),
        _meta(history=(0.48, 0.49, 0.50, 0.51)),
        path="cand_b",
    )
    assert b.aggregate_verdict == "fail"
    assert b.per_criterion["h2h_vs_v4_orig"].passed
    assert not b.per_criterion["h2h_vs_self_snapshot_floor"].passed


def test_select_winner_highest_v4_orig() -> None:
    """Two passing candidates with v4max 0.72 and 0.75 -> winner is the 0.75 one
    (highest ABSOLUTE anchor among passers)."""
    lo = _run_one(_per_opp(v4max=0.72), _meta(), path="cand_lo")
    hi = _run_one(_per_opp(v4max=0.75), _meta(), path="cand_hi")
    assert lo.aggregate_verdict == "pass"
    assert hi.aggregate_verdict == "pass"
    winner = select_e1_winner([lo, hi])
    assert winner is not None
    assert winner.candidate_path == "cand_hi"
    assert winner.h2h_vs_v4_orig == pytest.approx(0.75)


def test_no_passer_no_ship() -> None:
    """All candidates fail -> select_e1_winner returns None (no-ship verdict)."""
    fail = _run_one(_per_opp(v4max=0.68), _meta(), path="cand_fail")
    assert fail.aggregate_verdict == "fail"
    assert select_e1_winner([fail]) is None


def test_league_v5_compare_adaptive_strength_monotonicity_not_called() -> None:
    """Regression guard: e1_tournament.py does NOT import ``league_v5`` and does
    NOT CALL ``compare_adaptive_strength_monotonicity`` (the synthetic proxy,
    ``a_gate.py:33-36``). A docstring mention is allowed; an import or a call is
    not."""
    src = Path(__file__).resolve().parent.parent.joinpath("e1_tournament.py").read_text()
    # no import of league_v5 (any form).
    assert "import league_v5" not in src, "e1_tournament.py must NOT import league_v5"
    assert "from .league_v5" not in src, "e1_tournament.py must NOT import league_v5"
    assert "from train_v3.league_v5" not in src, (
        "e1_tournament.py must NOT import league_v5"
    )
    # no CALL of compare_adaptive_strength_monotonicity (the parenthesized call
    # form; a bare mention in a comment/docstring has no trailing '(').
    assert "compare_adaptive_strength_monotonicity(" not in src, (
        "e1_tournament.py must NOT call compare_adaptive_strength_monotonicity"
    )


def test_human_qa_is_soft_not_hard() -> None:
    """A candidate that PASSES the threshold table but has
    ``human_qa_verdict.verdict == 'easier'`` STILL PASSES (human-QA is SOFT per
    E-E8, NOT a hard gate; the hard ship decision is the threshold table)."""
    report = _run_one(
        _per_opp(),
        _meta(human_qa=_HumanQAFake(verdict="easier")),
    )
    assert report.aggregate_verdict == "pass", report.failed_criteria()
    assert report.human_qa_verdict.verdict == "easier"


def test_candidate_set_iteration_order_post_d_first() -> None:
    """The tournament processes post-D first, then post-C3, then post-B; Nones
    dropped; dedup -- asserted via the fake candidate_loader call sequence."""
    meta = _meta()
    runner = _FakeGameRunner(per_opponent_outcomes=_per_opp())
    loader = _FakeCandidateLoader(meta)
    config = E1TournamentConfig(
        candidate_set=E1CandidateSet(
            post_d_path="d",
            post_c3_best_path="c3",
            post_b_path="b",
        ),
        mana_draw_baseline=_baseline(),
        games_per_opponent=100,
        games_per_opponent_per_side=10,
        seeds=(0,),
    )
    reports = run_e1_tournament(
        config, game_runner=runner, candidate_loader=loader
    )
    assert loader.calls == ["d", "c3", "b"]
    assert [r.candidate_path for r in reports] == ["d", "c3", "b"]

    # Nones dropped + dedup: post_d='d', post_c3=None, post_b='d' (dup) -> ['d'].
    runner2 = _FakeGameRunner(per_opponent_outcomes=_per_opp())
    loader2 = _FakeCandidateLoader(meta)
    config2 = E1TournamentConfig(
        candidate_set=E1CandidateSet(post_d_path="d", post_c3_best_path=None, post_b_path="d"),
        mana_draw_baseline=_baseline(),
        games_per_opponent=100,
        games_per_opponent_per_side=10,
        seeds=(0,),
    )
    run_e1_tournament(config2, game_runner=runner2, candidate_loader=loader2)
    assert loader2.calls == ["d"], f"dedup+None-drop failed: {loader2.calls}"


def test_no_bonus_self_snapshot_hard_gate(tmp_path) -> None:
    """no_bonus p1/p2/second vs self-snapshot < 0.70 -> FAIL (the HARD gate); the
    V4-max advisory is a separate SOFT read (does not flip the verdict)."""
    # (1) HARD gate fails (no_bonus_p1 0.68 < 0.70) -> aggregate 'fail' even with
    #     a PASSING advisory JSON.
    adv_json = tmp_path / "adv.json"
    adv_json.write_text('{"no_bonus_p1": 0.80, "no_bonus_p2": 0.80, "no_bonus_second": 0.80}')
    config = E1TournamentConfig(
        candidate_set=E1CandidateSet(post_d_path="cand"),
        mana_draw_baseline=_baseline(),
        games_per_opponent=100,
        games_per_opponent_per_side=10,
        seeds=(0,),
        no_bonus_benchmark_json_path=str(adv_json),
    )
    report = _run_one(
        _per_opp(),
        _meta(no_bonus_p1=0.68),
        config=config,
    )
    assert report.aggregate_verdict == "fail"
    assert not report.per_criterion["no_bonus_p1"].passed
    # the advisory row passes but does NOT flip the verdict (SOFT).
    assert report.per_criterion["no_bonus_advisory_p1"].passed
    assert "no_bonus_p1" in report.failed_criteria()
    assert "no_bonus_advisory_p1" not in report.failed_criteria()

    # (2) HARD gate passes (no_bonus_p1 0.76 >= 0.70) but advisory FAILS
    #     (advisory p1 0.60 < 0.75) -> aggregate 'pass' (advisory is SOFT).
    adv_json2 = tmp_path / "adv2.json"
    adv_json2.write_text('{"no_bonus_p1": 0.60, "no_bonus_p2": 0.60, "no_bonus_second": 0.60}')
    config2 = E1TournamentConfig(
        candidate_set=E1CandidateSet(post_d_path="cand"),
        mana_draw_baseline=_baseline(),
        games_per_opponent=100,
        games_per_opponent_per_side=10,
        seeds=(0,),
        no_bonus_benchmark_json_path=str(adv_json2),
    )
    report2 = _run_one(_per_opp(), _meta(no_bonus_p1=0.76), config=config2)
    assert report2.aggregate_verdict == "pass", report2.failed_criteria()
    assert not report2.per_criterion["no_bonus_advisory_p1"].passed  # advisory fails
    assert report2.per_criterion["no_bonus_p1"].passed  # HARD passes
    # contract: failed_criteria() lists ONLY HARD rows -- a passing candidate
    # with failing SOFT advisories returns [] (empty iff passed).
    assert report2.failed_criteria() == []
    assert "no_bonus_advisory_p1" not in report2.failed_criteria()
    assert "no_bonus_advisory_p2" not in report2.failed_criteria()
    assert "no_bonus_advisory_second" not in report2.failed_criteria()


def test_regression_guard_no_edit_to_composed() -> None:
    """e1_tournament.py IMPORTS ``_check_h2h_vs_best`` from ``block_b_gate`` +
    does NOT redefine ``_check_h2h_vs_best`` / ``GateOutcome`` /
    ``ManaDrawBaseline`` (regression guard: E3 composes the A5/B6 pieces, it does
    NOT re-implement them)."""
    import train_v3.e1_tournament as mod
    src = Path(__file__).resolve().parent.parent.joinpath("e1_tournament.py").read_text()
    # imports _check_h2h_vs_best from block_b_gate.
    assert "from .block_b_gate import" in src
    assert "_check_h2h_vs_best" in src
    # does NOT redefine the composed types/functions (no `class GateOutcome` /
    # `class ManaDrawBaseline` / `def _check_h2h_vs_best` in the source).
    assert "class GateOutcome" not in src, "e1_tournament.py must NOT redefine GateOutcome"
    assert "class ManaDrawBaseline" not in src, (
        "e1_tournament.py must NOT redefine ManaDrawBaseline"
    )
    assert "def _check_h2h_vs_best" not in src, (
        "e1_tournament.py must NOT redefine _check_h2h_vs_best"
    )
    # the imported GateOutcome / ManaDrawBaseline are the A5 types (identity).
    from train_v3.a_gate import GateOutcome as A5GateOutcome
    from train_v3.a_gate import ManaDrawBaseline as A5Baseline
    assert mod.GateOutcome is A5GateOutcome
    assert mod.ManaDrawBaseline is A5Baseline


def test_unified_gauntlet_roster_shape() -> None:
    """E-E17: the unified roster is [v4max, random, end_turn, best_self_snapshot,
    *EXPLOIT_AGENT_KINDS] (11 lanes)."""
    assert UNIFIED_GAUNTLET_ROSTER == (
        "v4max",
        "random",
        "end_turn",
        "best_self_snapshot",
        *EXPLOIT_AGENT_KINDS,
    )
    assert len(UNIFIED_GAUNTLET_ROSTER) == 4 + 7


def test_threshold_constants_match_spec() -> None:
    """The module-level threshold constants match the SPEC :143-156 values."""
    assert H2H_V4_ORIG_MIN == 0.70
    assert H2H_VS_RANDOM_MIN == 0.95
    assert H2H_VS_END_TURN_MIN == 0.95
    assert H2H_VS_SELF_SNAPSHOT_MIN_FLOOR == 0.52
    assert P1_P2_MAX_SCORE_GAP == 0.12
    assert MIN_E2E_TPS == 12000.0
    assert MIN_ENTROPY == 0.70
    assert MAX_ABS_KL == 0.12
    assert NO_BONUS_P1_MIN == 0.70
    assert NO_BONUS_P2_MIN == 0.70
    assert NO_BONUS_SECOND_MIN == 0.70


def test_empty_candidate_set_returns_empty() -> None:
    """An all-None candidate set -> empty report list (no candidates to eval)."""
    config = E1TournamentConfig(
        candidate_set=E1CandidateSet(),
        mana_draw_baseline=_baseline(),
        games_per_opponent=100,
        games_per_opponent_per_side=10,
        seeds=(0,),
    )
    reports = run_e1_tournament(
        config,
        game_runner=_FakeGameRunner(per_opponent_outcomes=_per_opp()),
        candidate_loader=_FakeCandidateLoader(_meta()),
    )
    assert reports == []


def test_side_runner_measures_p1_p2_gap() -> None:
    """When a ``side_runner`` is supplied, the p1/p2 gap is MEASURED via
    ``second_start_parity.play_side_stratified_gauntlet`` (overrides the
    metadata p1_p2_gap). A side-stratified fake runner returning p1 wins / p2
    losses yields a large gap -> the p1_p2_gap HARD gate fails."""
    from train_v3.second_start_parity import BlockBGameResult

    class _FakeSideRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, str]] = []

        def play(self, opponent_kind: str, *, seed: int, candidate_side: str) -> BlockBGameResult:
            self.calls.append((opponent_kind, seed, candidate_side))
            outcome = "win" if candidate_side == "p1" else "loss"
            return BlockBGameResult(
                game=GameResult(
                    outcome=outcome,
                    mana_draw_count=0,
                    eligible_turns=1,
                    opponent=opponent_kind,
                ),
                candidate_side=candidate_side,
            )

    side = _FakeSideRunner()
    report = _run_one(
        _per_opp(),
        _meta(p1_p2_gap=0.0),  # metadata gap ignored when side_runner supplied
    )
    # re-run with side_runner injected (the _run_one helper doesn't expose it, so
    # build the tournament explicitly).
    runner = _FakeGameRunner(per_opponent_outcomes=_per_opp())
    loader = _FakeCandidateLoader(_meta(p1_p2_gap=0.0))
    config = E1TournamentConfig(
        candidate_set=E1CandidateSet(post_d_path="cand"),
        mana_draw_baseline=_baseline(),
        games_per_opponent=100,
        games_per_opponent_per_side=10,
        seeds=(0,),
    )
    reports = run_e1_tournament(
        config, game_runner=runner, candidate_loader=loader, side_runner=side
    )
    assert len(reports) == 1
    r = reports[0]
    # p1 all-win / p2 all-loss -> p1_rate=1.0, p2_rate=0.0 -> gap=1.0 > 0.12 -> FAIL.
    assert r.p1_p2_gap == pytest.approx(1.0)
    assert not r.per_criterion["p1_p2_gap"].passed
    assert r.aggregate_verdict == "fail"
    assert len(side.calls) > 0  # the side_runner was actually invoked


def test_mana_draw_baseline_required() -> None:
    """``E1TournamentConfig.mana_draw_baseline`` is REQUIRED -- a None baseline
    raises (the Q4 band gate is undefined without B)."""
    config = E1TournamentConfig(
        candidate_set=E1CandidateSet(post_d_path="cand"),
        mana_draw_baseline=None,  # type: ignore[arg-type]
        games_per_opponent=100,
        games_per_opponent_per_side=10,
        seeds=(0,),
    )
    with pytest.raises(ValueError, match="mana_draw_baseline"):
        run_e1_tournament(
            config,
            game_runner=_FakeGameRunner(per_opponent_outcomes=_per_opp()),
            candidate_loader=_FakeCandidateLoader(_meta()),
        )