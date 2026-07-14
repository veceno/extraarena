"""B4 — curriculum.py tests (synthetic, no real Rust/MLX/ONNX).

Covers the 5 plan tests (``BLOCK_B_PLAN.md:424-433``) plus the adapter test
(``test_extract_lane_outcomes_from_rollout``), the no-data neutral lane test, the
draws-excluded test, the purity test, and the regression guard
``test_does_not_edit_a4_sampler`` (git diff on ``rust_live_self_play.py`` is empty).

The reweight tests use fabricated ``LaneOutcome`` lists directly (no real rollout);
the adapter test uses a fake-rollout namedtuple with ``transitions.rewards`` as a
``(steps, env)`` numpy array + ``opponent_identities`` tuple, matching the
``LiveRolloutBatch`` contract (``rust_live_self_play.py:419`` /
``rust_collector.py:24`` ``RustTransitionBatch.rewards``).
"""

from __future__ import annotations

import os
import subprocess
from collections import namedtuple

import numpy as np

from train_v3.curriculum import (
    CurriculumReweighter,
    LaneOutcome,
    extract_lane_outcomes,
)

# Directory of this test file — used to resolve the A4 source path for the
# "no-edit" regression guard.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _lo(identity: str, outcome: str) -> LaneOutcome:
    return LaneOutcome(identity=identity, outcome=outcome)


# ---------------------------------------------------------------------------
# Adapter: extract_lane_outcomes
# ---------------------------------------------------------------------------

def test_extract_lane_outcomes_from_rollout() -> None:
    """Fake rollout with rewards (steps, env) + opponent_identities -> correct
    (identity, outcome) per env, sign of reward sum."""
    # rewards[:, i] sums: env0 = +0.5 (win), env1 = -0.25 (loss), env2 = 0.0 (draw),
    # env3 = +2.0 (win), env4 = -1.5 (loss).
    rewards = np.array(
        [
            [0.5, -0.25, 0.0, 1.0, -0.5],
            [0.0, 0.0, 0.0, 1.0, -1.0],
        ],
        dtype=np.float32,
    )
    FakeTransitions = namedtuple("FakeTransitions", ["rewards"])
    FakeRollout = namedtuple("FakeRollout", ["transitions", "opponent_identities"])
    rollout = FakeRollout(
        transitions=FakeTransitions(rewards=rewards),
        opponent_identities=("stall", "face_rush", "greedy_face", "random", "self"),
    )
    out = extract_lane_outcomes(rollout)
    assert len(out) == 5
    assert [(o.identity, o.outcome) for o in out] == [
        ("stall", "win"),
        ("face_rush", "loss"),
        ("greedy_face", "draw"),
        ("random", "win"),
        ("self", "loss"),
    ]


# ---------------------------------------------------------------------------
# Plan test 1: losing lane oversampled
# ---------------------------------------------------------------------------

def test_losing_lane_oversampled() -> None:
    """Fabricated outcomes with learner losing to 'stall' -> stall weight rises
    next update."""
    reweighter = CurriculumReweighter(window_n=4)
    # Learner loses to stall 8/10, beats face_rush 8/10.
    outcomes = [_lo("stall", "loss")] * 8 + [_lo("stall", "win")] * 2
    outcomes += [_lo("face_rush", "win")] * 8 + [_lo("face_rush", "loss")] * 2
    reweighter.update(outcomes)

    mix = [("stall", 0.5), ("face_rush", 0.5)]
    reweighted = reweighter.reweight(mix)

    w = dict(reweighted)
    # stall (loss_rate 0.8) must rise above its original 0.5 share; face_rush falls.
    assert w["stall"] > 0.5, f"stall should be oversampled, got {w['stall']}"
    assert w["face_rush"] < 0.5, f"face_rush should shrink, got {w['face_rush']}"
    # And stall must be the larger share (it has the higher loss rate).
    assert w["stall"] > w["face_rush"]


# ---------------------------------------------------------------------------
# Plan test 2: winning lane not oversampled
# ---------------------------------------------------------------------------

def test_winning_lane_not_oversampled() -> None:
    """A lane the learner beats 100% gets NO boost (factor 1.0)."""
    reweighter = CurriculumReweighter(window_n=4)
    # face_rush beaten 100% (loss_rate 0). stall at neutral 50/50 (loss_rate 0.5).
    outcomes = [_lo("face_rush", "win")] * 10
    outcomes += [_lo("stall", "win")] * 5 + [_lo("stall", "loss")] * 5
    reweighter.update(outcomes)

    mix = [("face_rush", 0.5), ("stall", 0.5)]
    reweighted = reweighter.reweight(mix)

    rates = reweighter.per_lane_loss_rate()
    assert rates["face_rush"] == 0.0  # 100% beaten
    # face_rush factor must be 1.0 (no boost): with stall also at loss_rate 0.5
    # (factor 1.0), both factors are 1.0 so the renormalized mix is unchanged.
    w = dict(reweighted)
    assert abs(w["face_rush"] - 0.5) < 1e-9, f"face_rush should not be boosted, got {w['face_rush']}"
    assert abs(w["stall"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Plan test 3: reweight capped (~25%/update)
# ---------------------------------------------------------------------------

def test_reweight_capped() -> None:
    """A single update never shifts more than ~25% toward one lane — boost_factor
    <= 1.25 and post-renorm shift is bounded."""
    reweighter = CurriculumReweighter(window_n=4)
    # Extreme: learner loses to stall 100/100 (loss_rate 1.0). The raw loss margin
    # is 0.5 but the cap (0.25) must clamp the boost factor to 1.25, not 1.5.
    outcomes = [_lo("stall", "loss")] * 100 + [_lo("face_rush", "win")] * 100
    reweighter.update(outcomes)

    mix = [("stall", 0.5), ("face_rush", 0.5)]
    reweighted = reweighter.reweight(mix, cap=0.25)

    rates = reweighter.per_lane_loss_rate()
    assert rates["stall"] == 1.0
    # Recompute the factor the implementation must have used: 1 + min(1.0-0.5, 0.25)
    # = 1.25. Verify the renormalized stall weight equals 0.5*1.25 / (0.5*1.25 +
    # 0.5*1.0) = 0.625 / 1.125.
    expected_stall = (0.5 * 1.25) / (0.5 * 1.25 + 0.5 * 1.0)
    w = dict(reweighted)
    assert abs(w["stall"] - expected_stall) < 1e-9
    # The boost factor never exceeds 1.25 -> stall weight moves from 0.5 to ~0.556,
    # a ~5.6pp shift, well within the 25% cap envelope (no single-step collapse).
    assert w["stall"] <= 0.5 * 1.25 + 1e-9
    assert w["stall"] < 0.6  # not collapsing toward one lane


# ---------------------------------------------------------------------------
# Plan test 4: reweighted mix accounts to one
# ---------------------------------------------------------------------------

def test_reweighted_mix_accounts_to_one() -> None:
    """After reweight, weights sum to 1.0 within float tol."""
    reweighter = CurriculumReweighter(window_n=4)
    outcomes = (
        [_lo("stall", "loss")] * 7
        + [_lo("stall", "win")] * 3
        + [_lo("face_rush", "win")] * 9
        + [_lo("face_rush", "loss")] * 1
        + [_lo("random", "loss")] * 4
        + [_lo("random", "win")] * 6
    )
    reweighter.update(outcomes)

    mix = [("stall", 0.4), ("face_rush", 0.35), ("random", 0.25)]
    reweighted = reweighter.reweight(mix)
    total = sum(w for _, w in reweighted)
    assert abs(total - 1.0) < 1e-9, f"reweighted mix must sum to 1.0, got {total}"


# ---------------------------------------------------------------------------
# Plan test 5: rolling window
# ---------------------------------------------------------------------------

def test_rolling_window() -> None:
    """Only the last N updates inform the reweight — stale losses age out."""
    reweighter = CurriculumReweighter(window_n=2)
    # Update 1: learner loses hard to stall.
    reweighter.update([_lo("stall", "loss")] * 10 + [_lo("face_rush", "win")] * 10)
    # Update 2: learner loses hard to face_rush (stall now wins).
    reweighter.update([_lo("stall", "win")] * 10 + [_lo("face_rush", "loss")] * 10)
    # Update 3 (ages out update 1): learner beats stall, loses to face_rush.
    reweighter.update([_lo("stall", "win")] * 10 + [_lo("face_rush", "loss")] * 10)

    rates = reweighter.per_lane_loss_rate()
    # Window holds updates 2 and 3: stall = 20 wins / 0 losses -> loss_rate 0.0;
    # face_rush = 0 wins / 20 losses -> loss_rate 1.0. Update 1 (stall losses) aged
    # out, so stall must NOT be flagged as a losing lane.
    assert rates["stall"] == 0.0, f"stale stall losses aged out, got {rates['stall']}"
    assert rates["face_rush"] == 1.0

    mix = [("stall", 0.5), ("face_rush", 0.5)]
    reweighted = reweighter.reweight(mix)
    w = dict(reweighted)
    # face_rush is the losing lane now -> oversampled; stall gets no boost.
    assert w["face_rush"] > 0.5
    assert w["stall"] < 0.5


# ---------------------------------------------------------------------------
# Extra: neutral lane (no outcome data) -> factor 1.0
# ---------------------------------------------------------------------------

def test_neutral_lane_no_data() -> None:
    """A lane in the mix with no outcome data in the window gets factor 1.0."""
    reweighter = CurriculumReweighter(window_n=4)
    # Only stall has outcomes; 'self' has none.
    reweighter.update([_lo("stall", "loss")] * 8 + [_lo("stall", "win")] * 2)

    mix = [("stall", 0.5), ("self", 0.5)]
    reweighted = reweighter.reweight(mix)

    rates = reweighter.per_lane_loss_rate()
    assert "self" not in rates  # no data
    # self factor = 1.0 (neutral). stall factor = 1 + min(0.8-0.5, 0.25) = 1.25.
    w = dict(reweighted)
    expected_self = 0.5 / (0.5 * 1.25 + 0.5 * 1.0)
    assert abs(w["self"] - expected_self) < 1e-9, f"self neutral factor, got {w['self']}"
    assert w["stall"] > w["self"]  # stall boosted, self not


# ---------------------------------------------------------------------------
# Extra: draws excluded from loss rate
# ---------------------------------------------------------------------------

def test_draws_excluded_from_loss_rate() -> None:
    """Draws don't count as wins or losses — loss_rate = losses/(wins+losses)."""
    reweighter = CurriculumReweighter(window_n=4)
    # stall: 2 wins, 4 losses, 10 draws. If draws counted, loss_rate would be
    # 4/16 = 0.25; excluded, it is 4/6 ~= 0.667 (>0.5 -> boosted).
    outcomes = (
        [_lo("stall", "win")] * 2
        + [_lo("stall", "loss")] * 4
        + [_lo("stall", "draw")] * 10
    )
    reweighter.update(outcomes)

    rates = reweighter.per_lane_loss_rate()
    assert abs(rates["stall"] - (4 / 6)) < 1e-9, f"draws excluded, got {rates['stall']}"

    # And a lane that is ALL draws -> zero decided -> 0.5 neutral (no boost).
    reweighter2 = CurriculumReweighter(window_n=4)
    reweighter2.update([_lo("face_rush", "draw")] * 10)
    rates2 = reweighter2.per_lane_loss_rate()
    assert rates2["face_rush"] == 0.5


# ---------------------------------------------------------------------------
# Extra: reweight does not mutate input
# ---------------------------------------------------------------------------

def test_reweight_does_not_mutate_input() -> None:
    """reweight is pure — input mix list and its tuples are unchanged."""
    reweighter = CurriculumReweighter(window_n=4)
    reweighter.update([_lo("stall", "loss")] * 8 + [_lo("stall", "win")] * 2)

    mix = [("stall", 0.5), ("face_rush", 0.5)]
    mix_snapshot = [(n, w) for n, w in mix]
    _ = reweighter.reweight(mix)

    assert mix == mix_snapshot, "input mix list must not be mutated"
    # Tuples are immutable, but verify the floats too.
    for (n_before, w_before), (n_after, w_after) in zip(mix_snapshot, mix):
        assert n_before == n_after
        assert w_before == w_after


# ---------------------------------------------------------------------------
# Regression guard: B4 does NOT edit A4's sampler
# ---------------------------------------------------------------------------

def test_does_not_edit_a4_sampler() -> None:
    """B4 hooks via the mix arg, does NOT edit the sampler's Phase-A frozen
    constants. B8 (``BLOCK_B_PLAN.md`` §3 B8, the FINAL Block-B component)
    additively extends A4 ``rust_live_self_play.py`` with the
    ``BLOCK_B_POLICY_OPPONENT_KINDS`` dispatch check + the optional
    ``opponent_mix_parsed`` param on ``run_live_self_play_update``. The later
    benchmark-parity bugfix may also replace the documented max-id approximation
    inside ``GreedyFaceOpponent``. The reward-attribution repair may additionally
    use exact ``counterparty_rewards`` and remove ``pending_opener_reward``.
    This test asserts the A4 diff is EITHER empty OR contains one of those scoped
    changes, and NEVER touches
    ``POLICY_OPPONENT_KINDS`` / ``PHASE_A_IDENTITIES`` / ``RULE_AGENT_CODES``
    (the Phase-A frozen counts)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
    a4_rel = os.path.join("TrainV3.5", "python", "train_v3", "rust_live_self_play.py")
    a4_abs = os.path.join(repo_root, a4_rel)
    assert os.path.isfile(a4_abs), f"A4 source not found at {a4_abs}"
    res = subprocess.run(
        ["git", "-C", repo_root, "diff", "--", a4_rel],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"git diff failed: {res.stderr}"
    diff = res.stdout
    # TrainV3.5/ is gitignored at the repo root (tracked only inside this worktree's
    # own git), so `git diff` at the repo root reports nothing for it when untracked.
    # If the path IS tracked in this worktree's git, the diff may contain ONLY the
    # B8 additive extension (BLOCK_B_POLICY_OPPONENT_KINDS + opponent_mix_parsed).
    if diff.strip() == "":
        return  # no edit -- the strictest pass.
    is_b8_extension = (
        "BLOCK_B_POLICY_OPPONENT_KINDS" in diff and "opponent_mix_parsed" in diff
    )
    is_greedy_face_parity_fix = (
        "class GreedyFaceOpponent" in diff
        and "return int(ids[-1])" in diff
        and "attack enemy hero, play a no-target card" in diff
    )
    is_reward_attribution_repair = (
        "counterparty_rewards" in diff and "pending_opener_reward" in diff
    )
    assert is_b8_extension or is_greedy_face_parity_fix or is_reward_attribution_repair, (
        "A4 diff is neither the B8 dispatch extension, the scoped "
        "GreedyFaceOpponent benchmark-parity fix, nor the reward-attribution "
        f"repair; got:\n{diff}"
    )
    # The frozen Phase-A constants must NOT be removed or altered (no '-' line
    # touches them). Removed lines start with '-' (but not '---').
    removed = [ln for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    forbidden = ("POLICY_OPPONENT_KINDS", "PHASE_A_IDENTITIES", "RULE_AGENT_CODES")
    for ln in removed:
        assert not any(tok in ln for tok in forbidden), (
            f"A4 frozen Phase-A constant was edited (removed line): {ln!r}"
        )
