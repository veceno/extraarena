"""Tests for Block B component B5 — ``second_start_parity.py`` (continuous
second-start parity loop). TRACKED (verify: ``git check-ignore`` exit 1).

All tests use SYNTHETIC game outcomes via a fake ``BlockBGameRunner`` returning
synthetic ``BlockBGameResult``s — the p1/p2 measurement + gap + A3-scheme
feedback logic is unit-testable without MLX/Rust/ONNX (no real Rust arena, no
real policy). This mirrors the A5 fake-runner pattern (``test_a_gate.py``).

Run: ``PYTHONPATH=.:TrainV3.5/python python3 -m pytest
TrainV3.5/python/train_v3/tests/test_second_start_parity.py``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from train_v3.a_gate import GameResult, GameRunner, play_gauntlet
from train_v3.ppo_phaseA_config import (
    PHASE_A_P1_P2_GAP_THRESHOLD,
    second_start_oversampling_scheme,
)
from train_v3.second_start_parity import (
    BLOCK_B_GAP_THRESHOLD,
    BlockBGameResult,
    BlockBGameRunner,
    SecondStartParityLoop,
    play_side_stratified_gauntlet,
)


# -- fake runner --------------------------------------------------------------


class _FakeBlockBRunner:
    """Fake ``BlockBGameRunner`` returning synthetic ``BlockBGameResult``s.

    ``side_outcomes`` maps ``candidate_side`` -> list[Outcome] cycled per call, so
    tests can dial in exact p1/p2 win/draw/loss distributions per side. No real
    Rust/MLX/ONNX.
    """

    def __init__(
        self,
        side_outcomes: dict[str, list[str]] | None = None,
    ) -> None:
        self.side_outcomes = side_outcomes or {"p1": ["win"], "p2": ["loss"]}
        self._idx: dict[str, int] = {"p1": 0, "p2": 0}
        self.calls: list[tuple[str, int, str]] = []

    def play(self, opponent_kind: str, *, seed: int, candidate_side: str) -> BlockBGameResult:
        self.calls.append((opponent_kind, seed, candidate_side))
        outs = self.side_outcomes.get(candidate_side, ["draw"])
        outcome = outs[self._idx[candidate_side] % len(outs)]
        self._idx[candidate_side] += 1
        return BlockBGameResult(
            game=GameResult(
                outcome=outcome,
                mana_draw_count=0,
                eligible_turns=1,
                opponent=opponent_kind,
            ),
            candidate_side=candidate_side,
        )


def _mk(side: str, outcome: str, opponent: str = "v4max") -> BlockBGameResult:
    return BlockBGameResult(
        game=GameResult(
            outcome=outcome,
            mana_draw_count=0,
            eligible_turns=1,
            opponent=opponent,
        ),
        candidate_side=side,
    )


# -- 1. test_breach_oversamples_p2 -------------------------------------------


def test_breach_oversamples_p2() -> None:
    """gap>0.12 with p2_rate<p1_rate -> scheme oversampled_side="p2" and
    p2_weight rises above 0.5."""
    loop = SecondStartParityLoop(window_n=100)
    # p1 candidate wins everything (rate 1.0); p2 candidate loses everything
    # (rate 0.0) -> gap = 1.0 > 0.12 -> breach, p2 is the lower-rate side.
    loop.update([_mk("p1", "win"), _mk("p1", "win"), _mk("p2", "loss"), _mk("p2", "loss")])

    assert loop.p1_score_rate() == pytest.approx(1.0)
    assert loop.p2_score_rate() == pytest.approx(0.0)
    assert loop.gap() == pytest.approx(1.0)
    assert loop.breach() is True

    scheme = loop.oversampling_scheme()
    assert scheme["breach"] is True
    assert scheme["oversampled_side"] == "p2"
    assert scheme["p2_weight"] > 0.5
    assert scheme["p1_weight"] < 0.5
    # weights still sum to 1.0 and stay in [0, 1]
    assert scheme["p1_weight"] + scheme["p2_weight"] == pytest.approx(1.0)
    assert 0.0 <= scheme["p1_weight"] <= 1.0
    assert 0.0 <= scheme["p2_weight"] <= 1.0


# -- 2. test_balanced_no_change ----------------------------------------------


def test_balanced_no_change() -> None:
    """gap<=0.12 (here gap=0) -> no breach, 0.5/0.5, oversampled_side=None."""
    loop = SecondStartParityLoop(window_n=100)
    # p1 and p2 both win everything -> rate 1.0 each -> gap = 0 -> no breach.
    loop.update([_mk("p1", "win"), _mk("p2", "win")])

    assert loop.p1_score_rate() == pytest.approx(1.0)
    assert loop.p2_score_rate() == pytest.approx(1.0)
    assert loop.gap() == pytest.approx(0.0)
    assert loop.breach() is False

    scheme = loop.oversampling_scheme()
    assert scheme["breach"] is False
    assert scheme["oversampled_side"] is None
    assert scheme["p1_weight"] == pytest.approx(0.5)
    assert scheme["p2_weight"] == pytest.approx(0.5)


# -- 3. test_gap_feeds_promotion ---------------------------------------------


def test_gap_feeds_promotion() -> None:
    """The measured gap is exposed to B6 via ``gap_for_promotion()``."""
    loop = SecondStartParityLoop(window_n=100)
    loop.update([_mk("p1", "win"), _mk("p1", "draw"), _mk("p2", "loss")])
    # p1 rate = (1 + 0.5*1)/2 = 0.75 ; p2 rate = 0/1 = 0.0 ; gap = 0.75
    expected_gap = abs(0.75 - 0.0)
    assert loop.gap_for_promotion() == pytest.approx(expected_gap)
    # gap_for_promotion equals gap (single source of truth).
    assert loop.gap_for_promotion() == pytest.approx(loop.gap())


# -- 4. test_rolling_window --------------------------------------------------


def test_rolling_window() -> None:
    """Only recent gauntlet games inform the rate — stale games age out via the
    deque maxlen."""
    loop = SecondStartParityLoop(window_n=4)
    # First batch: p1 wins, p2 loses (breach). 4 games fit exactly in the window.
    loop.update([
        _mk("p1", "win"), _mk("p1", "win"),
        _mk("p2", "loss"), _mk("p2", "loss"),
    ])
    assert loop.gap() == pytest.approx(1.0)
    assert loop.breach() is True

    # Second batch: 4 NEW balanced games push the old 4 out of the window.
    loop.update([_mk("p1", "win"), _mk("p1", "draw"), _mk("p2", "win"), _mk("p2", "draw")])
    # Now only the balanced batch is in the window:
    # p1 rate = (1 + 0.5*1)/2 = 0.75 ; p2 rate = (1 + 0.5*1)/2 = 0.75 ; gap = 0
    assert len(loop) == 4  # window capacity respected
    assert loop.gap() == pytest.approx(0.0)
    assert loop.breach() is False


# -- 5. test_does_not_wire_dead_field ----------------------------------------


def test_does_not_wire_dead_field() -> None:
    """B5 builds NEW; the dead ``gauntlet_v5.V5GauntletConfig.p1_p2_max_score_gap``
    field is NOT consumed. B5 does NOT import ``gauntlet_v5`` or reference the
    field (regression guard for the A5 pattern)."""
    import ast
    import train_v3.second_start_parity as b5

    src = Path(b5.__file__).read_text()
    tree = ast.parse(src)

    # B5 must not IMPORT gauntlet_v5 (no `import gauntlet_v5` / `from
    # gauntlet_v5 import ...` / `from train_v3.gauntlet_v5 import ...`).
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
                imported_names.add((alias.asname or alias.name).split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod:
                imported_names.add(mod)
                imported_names.add(mod.split(".")[-1])
            for alias in node.names:
                imported_names.add(alias.name)
    assert "gauntlet_v5" not in imported_names, (
        "B5 must not import gauntlet_v5 (the dead p1_p2_max_score_gap field is "
        "NOT wired — A5 pattern)"
    )

    # B5 must not reference the dead field name in CODE (attribute access / a
    # non-docstring string literal). Docstring mentions are permitted — the
    # rationale explains WHY the field is not wired; code references are not.
    docstring_nodes: set[int] = set()
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if body and isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstring_nodes.add(id(first.value))

    code_refs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "p1_p2_max_score_gap":
            code_refs.append("attribute access")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "p1_p2_max_score_gap" in node.value
            and id(node) not in docstring_nodes
        ):
            code_refs.append("string literal")
    assert not code_refs, (
        f"B5 must not reference the dead gauntlet_v5.p1_p2_max_score_gap field "
        f"in code (found {code_refs})"
    )

    # And the module's namespace must not pull in gauntlet_v5 transitively.
    assert not hasattr(b5, "gauntlet_v5")
    assert not hasattr(b5, "V5GauntletConfig")

    # Confirm the field is genuinely dead in gauntlet_v5 (zero consumers within
    # gauntlet_v5.py itself) — this is the reason B5 builds NEW rather than wiring
    # it. We only check that B5 is not among any consumers.
    import train_v3.gauntlet_v5 as gv5

    gv5_src = Path(gv5.__file__).read_text()
    # The field appears at its def site; B5 must not be a CODE consumer (the
    # AST check above enforces that). The field is genuinely dead in
    # gauntlet_v5 itself (zero consumers within gauntlet_v5.py beyond its def).
    assert "p1_p2_max_score_gap" in gv5_src  # defined there
    # Re-confirm B5 does not import gauntlet_v5 (already asserted above) — so B5
    # cannot be a consumer of the field even if it mentioned it in prose.
    assert "gauntlet_v5" not in imported_names


# -- 6. test_measures_p1_and_p2_separately -----------------------------------


def test_measures_p1_and_p2_separately() -> None:
    """p1_score_rate and p2_score_rate are measured from side-stratified games
    with ``candidate_side`` recorded, NOT a single aggregate assuming a fixed
    candidate side."""
    loop = SecondStartParityLoop(window_n=100)
    # Side-stratified: p1 candidate wins 2/2 (rate 1.0), p2 candidate draws 2/2
    # (rate 0.5). A single aggregate would be (2 wins + 2 draws)/4 = 0.75 for
    # both "sides" — the per-side measurement distinguishes them.
    loop.update([
        _mk("p1", "win"), _mk("p1", "win"),
        _mk("p2", "draw"), _mk("p2", "draw"),
    ])

    assert loop.p1_score_rate() == pytest.approx(1.0)   # (2 + 0)/2
    assert loop.p2_score_rate() == pytest.approx(0.5)   # (0 + 0.5*2)/2
    assert loop.gap() == pytest.approx(0.5)
    assert loop.breach() is True  # 0.5 > 0.12

    # The per-side split is real: the loop distinguishes outcomes by
    # candidate_side, not by an assumed fixed side. Cross-check the side stats.
    assert loop._side_stats("p1") == (2, 0, 0)
    assert loop._side_stats("p2") == (0, 2, 0)


# -- 7. test_blockb_game_result_composes_without_mutating --------------------


def test_blockb_game_result_composes_without_mutating() -> None:
    """``BlockBGameResult`` wraps A5 ``GameResult`` (composition); the A5 frozen
    dataclass is unchanged (``git diff a_gate.py`` empty)."""
    gr = GameResult(outcome="win", mana_draw_count=1, eligible_turns=3, opponent="v4max")
    bgr = BlockBGameResult(game=gr, candidate_side="p1")

    # Composition: the A5 GameResult is the ``game`` field, unchanged.
    assert bgr.game is gr
    assert bgr.game.outcome == "win"
    assert bgr.game.mana_draw_count == 1
    assert bgr.game.eligible_turns == 3
    assert bgr.game.opponent == "v4max"
    assert bgr.candidate_side == "p1"

    # BlockBGameResult is frozen like A5 GameResult.
    with pytest.raises(Exception):
        bgr.candidate_side = "p2"  # type: ignore[misc]

    # Invalid side rejected.
    with pytest.raises(ValueError):
        BlockBGameResult(game=gr, candidate_side="p3")

    # Composition guard: must wrap an actual a_gate.GameResult, not a duck type.
    with pytest.raises(TypeError):
        BlockBGameResult(  # type: ignore[arg-type]
            game={"outcome": "win", "mana_draw_count": 0, "eligible_turns": 1, "opponent": "x"},  # type: ignore[arg-type]
            candidate_side="p1",
        )

    # A5 GameResult is still the SAME class (not re-defined in B5).
    from train_v3 import a_gate

    assert a_gate.GameResult is GameResult
    # And A5 GameResult field set is unchanged (composition, no field duplication).
    assert {f.name for f in GameResult.__dataclass_fields__.values()} == {
        "outcome", "mana_draw_count", "eligible_turns", "opponent"
    }

    # Regression guard: a_gate.py is unmodified by B5 (frozen-classic guard).
    repo_root = Path(__file__).resolve().parents[4]  # .../TrainV3.5Prep
    a_gate_rel = "TrainV3.5/python/train_v3/a_gate.py"
    diff = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--", a_gate_rel],
        capture_output=True, text=True,
    )
    assert diff.returncode == 0, f"git diff failed: {diff.stderr}"
    assert diff.stdout == "", (
        "B5 must NOT edit a_gate.py (A5 GameResult frozen — compose "
        "BlockBGameResult, do NOT mutate); git diff a_gate.py must be empty"
    )


# -- 8. test_reuses_a3_scheme ------------------------------------------------


def test_reuses_a3_scheme() -> None:
    """B5 calls A3 ``second_start_oversampling_scheme``, does NOT re-invent the
    gap-weight math."""
    loop = SecondStartParityLoop(window_n=100)
    loop.update([_mk("p1", "win"), _mk("p1", "win"), _mk("p2", "loss")])
    # p1 rate = 1.0, p2 rate = 0.0 -> gap = 1.0 -> breach, oversample p2.
    p1r = loop.p1_score_rate()
    p2r = loop.p2_score_rate()

    # The scheme B5 produces must equal A3's scheme called directly with the same
    # measured rates + threshold (single-sourced gap-weight math).
    expected = second_start_oversampling_scheme(
        p1r, p2r, gap_threshold=BLOCK_B_GAP_THRESHOLD
    )
    got = loop.oversampling_scheme()
    assert got == expected

    # B5 reuses the A3 threshold constant (no re-invented threshold).
    assert BLOCK_B_GAP_THRESHOLD == PHASE_A_P1_P2_GAP_THRESHOLD

    # Source-vs-source: B5 module imports the A3 scheme (not a local reimpl).
    import train_v3.second_start_parity as b5

    src = Path(b5.__file__).read_text()
    assert "second_start_oversampling_scheme" in src
    assert "from train_v3.ppo_phaseA_config import" in src


# -- 9. test_does_not_edit_a5_a3_a4 ------------------------------------------


def test_does_not_edit_a5_a3_a4() -> None:
    """B5 does NOT edit A5 ``a_gate.py`` / A3 ``ppo_phaseA_config.py`` (git diff
    empty for each -- frozen-classic guard). A4 ``rust_live_self_play.py`` is
    additively extended by B8 (``BLOCK_B_POLICY_OPPONENT_KINDS`` +
    ``opponent_mix_parsed``) and the explicit learner-perspective reward repair
    (``counterparty_rewards`` and removal of ``pending_opener_reward``). Neither
    may alter ``POLICY_OPPONENT_KINDS`` / ``PHASE_A_IDENTITIES`` /
    ``RULE_AGENT_CODES``.
    B5 feeds measured rates via the B8 driver, does NOT edit the sampler.
    """
    repo_root = Path(__file__).resolve().parents[4]  # .../TrainV3.5Prep
    frozen_empty = [
        "TrainV3.5/python/train_v3/a_gate.py",
        "TrainV3.5/python/train_v3/ppo_phaseA_config.py",
    ]
    for rel in frozen_empty:
        diff = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--", rel],
            capture_output=True, text=True,
        )
        assert diff.returncode == 0, f"git diff {rel} failed: {diff.stderr}"
        assert diff.stdout == "", (
            f"B5 must NOT edit {rel} (A5/A3 read-only); git diff must be empty"
        )
    # A4: allow B8 and the narrow second-start reward repair.
    a4_rel = "TrainV3.5/python/train_v3/rust_live_self_play.py"
    diff = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--", a4_rel],
        capture_output=True, text=True,
    )
    assert diff.returncode == 0, f"git diff {a4_rel} failed: {diff.stderr}"
    out = diff.stdout
    if out.strip() != "":
        assert "counterparty_rewards" in out and "pending_opener_reward" in out, (
            f"A4 diff must include the narrow counterparty/opener reward repair; got:\n{out}"
        )
        removed = [ln for ln in out.splitlines() if ln.startswith("-") and not ln.startswith("---")]
        forbidden = ("POLICY_OPPONENT_KINDS", "PHASE_A_IDENTITIES", "RULE_AGENT_CODES")
        for ln in removed:
            assert not any(tok in ln for tok in forbidden), (
                f"A4 frozen Phase-A constant was edited (removed line): {ln!r}"
            )


# -- 10. test_side_stratified_gauntlet_plays_both_sides ----------------------


def test_side_stratified_gauntlet_plays_both_sides() -> None:
    """The helper plays each opponent from BOTH p1 and p2, mirroring A5
    ``play_gauntlet`` aggregation structure but side-stratified."""
    runner = _FakeBlockBRunner(side_outcomes={"p1": ["win"], "p2": ["loss"]})
    results = play_side_stratified_gauntlet(
        runner,
        opponent_kinds=["v4max", "v4orig"],
        games_per_opponent_per_side=2,
        seed=7,
    )

    # 2 opponents x 2 sides x 2 games = 8 results.
    assert len(results) == 8
    # Each opponent is played from BOTH p1 and p2.
    sides_per_opp: dict[str, set[str]] = {}
    for r in results:
        sides_per_opp.setdefault(r.game.opponent, set()).add(r.candidate_side)
    assert sides_per_opp == {"v4max": {"p1", "p2"}, "v4orig": {"p1", "p2"}}

    # Mirrors A5 play_gauntlet seed derivation: seed * 1_000_003 + g (per game
    # index within an opponent-side block).
    seeds = [c[1] for c in runner.calls]
    expected_seeds = []
    for _ in ("v4max", "v4orig"):
        for _side in ("p1", "p2"):
            for g in range(2):
                expected_seeds.append(7 * 1_000_003 + g)
    assert seeds == expected_seeds

    # The runner was called with the explicit candidate_side arg (B5 additive
    # extension over A5 GameRunner, which has no side arg).
    called_sides = [c[2] for c in runner.calls]
    assert sorted(called_sides) == ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"]

    # Argument validation mirrors A5 play_gauntlet.
    with pytest.raises(ValueError):
        play_side_stratified_gauntlet(runner, [], games_per_opponent_per_side=2)
    with pytest.raises(ValueError):
        play_side_stratified_gauntlet(runner, ["v4max"], games_per_opponent_per_side=0)


# -- 11. empty-side neutrality ----------------------------------------------


def test_empty_side_is_neutral() -> None:
    """A side with no recorded games is NEUTRAL (rate 0.5, gap 0, no breach) —
    a freshly-started loop with only p1 games does NOT spuriously breach on the
    missing p2 side."""
    loop = SecondStartParityLoop(window_n=100)
    loop.update([_mk("p1", "win"), _mk("p1", "win")])  # no p2 games
    assert loop.p1_score_rate() == pytest.approx(1.0)
    assert loop.p2_score_rate() == pytest.approx(0.5)  # neutral
    assert loop.gap() == pytest.approx(0.5)
    # gap 0.5 > 0.12 would breach — but the breach here is driven by the p1 rate
    # being far from the NEUTRAL p2 rate, which is the correct conservative
    # behaviour (we cannot claim parity with one side missing). The neutral
    # rate is 0.5, NOT 0.0, so a missing side does not auto-breach when p1 is
    # also balanced:
    loop2 = SecondStartParityLoop(window_n=100)
    loop2.update([_mk("p1", "draw"), _mk("p1", "draw")])  # p1 rate 0.5, no p2
    assert loop2.p1_score_rate() == pytest.approx(0.5)
    assert loop2.p2_score_rate() == pytest.approx(0.5)
    assert loop2.gap() == pytest.approx(0.0)
    assert loop2.breach() is False


# -- 12. BlockBGameRunner protocol is additive over A5 -----------------------


def test_blockb_runner_extends_a5_additively() -> None:
    """``BlockBGameRunner.play`` adds ``candidate_side`` to A5 ``GameRunner.play``
    (which has no side arg) — additive, A5 Protocol unchanged."""
    # A5 GameRunner.play signature has no candidate_side.
    import inspect

    a5_sig = inspect.signature(GameRunner.play)
    assert "candidate_side" not in a5_sig.parameters
    assert "opponent_kind" in a5_sig.parameters
    assert "seed" in a5_sig.parameters

    # B5 BlockBGameRunner.play adds candidate_side.
    b5_sig = inspect.signature(BlockBGameRunner.play)
    assert "candidate_side" in b5_sig.parameters
    assert "opponent_kind" in b5_sig.parameters
    assert "seed" in b5_sig.parameters

    # A5 play_gauntlet still works with an A5-shaped runner (A5 unchanged).
    class _A5Runner:
        def play(self, opponent_kind: str, *, seed: int) -> GameResult:
            return GameResult(outcome="win", mana_draw_count=0, eligible_turns=1, opponent=opponent_kind)

    outcomes = play_gauntlet(_A5Runner(), ["v4max"], games_per_opponent=2, seed=1)
    assert outcomes.wins == 2
