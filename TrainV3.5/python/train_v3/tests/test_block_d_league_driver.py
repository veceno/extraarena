"""Tests for Block D component D2 -- ``block_d_league_driver.py`` (NEW).

The Block-D post-C consolidation league driver: a SUBCLASS of B8
``BlockBLeagueDriver`` overriding ``_build_reweighted_mix`` (D1 + merge) +
``run()`` (D-D3 fixed-schedule / plateau exit + D->E1 handoff). Synthetic tests
only -- ``FakeWorker`` (reused from the A4 test module) + a fake
``BlockBGameRunner`` + fake ``opponent_policies``. The PPO optimizer step is
MLX-gated inside A4 (skipped when ``model`` / ``optimizer`` are None); the
league loop itself runs without MLX.

Run:
  PYTHONPATH=.:TrainV3.5/python python3 -m pytest \\
      TrainV3.5/python/train_v3/tests/test_block_d_league_driver.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

# Reuse the A4 FakeWorker / _FakeLearner / _EndTurnPolicy / script builder so the
# D2 tests drive A4 ``collect_rust_live_rollout`` with the SAME deterministic
# stand-in the A4 + B8 suites use (source-vs-source: A4 = oracle, B8 = oracle,
# D2 = UUT). The tests dir has no __init__.py; add it to sys.path so the sibling
# test modules are importable as top-level modules.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_rust_live_self_play as a4test  # noqa: E402
import test_block_b_league_driver as b8test  # noqa: E402

FakeWorker = a4test.FakeWorker
_FakeLearner = a4test._FakeLearner
_EndTurnPolicy = a4test._EndTurnPolicy
_script_alternating = a4test._script_alternating

# Reuse the B8 fake game runner + league config + worker factory (stable, tested
# helpers -- the D2 loop is a copy of the B8 loop, so the same fakes drive it).
_FakeGameRunner = b8test._FakeGameRunner
_league_config = b8test._league_config
_worker_factory = b8test._worker_factory

import train_v3.block_b_league_driver as b8mod  # noqa: E402
import train_v3.block_d_league_driver as d2mod  # noqa: E402
from train_v3 import rust_live_self_play as rls  # noqa: E402
from train_v3.a_gate import GameResult, ManaDrawBaseline  # noqa: E402
from train_v3.block_b_gate import DEFAULT_BLOCK_B_N_SNAP  # noqa: E402
from train_v3.c_to_d_handoff import E1CandidateSet  # noqa: E402
from train_v3.curriculum import CurriculumReweighter  # noqa: E402
from train_v3.exit_to_c2 import DEFAULT_K_SNAP, DEFAULT_MIN_GAIN  # noqa: E402
from train_v3.ppo_phaseA_config import PhaseAPPOConfig  # noqa: E402
from train_v3.second_start_parity import (  # noqa: E402
    BlockBGameResult,
    SecondStartParityLoop,
)
from train_v3.snapshot_pool import SnapshotEntry, SnapshotPool  # noqa: E402
from train_v3.v4_orig_temp_spectrum import TempV4Opponent  # noqa: E402


# =============================================================================
# Fake opponent_policies factory -- wires EVERY policy-dispatch identity the D1
# mix can emit (after the v5_snapshot->self merge): self / greedy_face /
# end_turn / v4max (POLICY_OPPONENT_KINDS) + v4-orig-argmax / t07 / t12
# (BLOCK_B_POLICY_OPPONENT_KINDS). Rule-agent identities (stall /
# anti_draw_greed / punish_empty_board / random) dispatch via Rust rule codes
# (FakeWorker.select_rule_actions) -- NO policy entry needed. NOTE: B8's factory
# omits ``end_turn`` (the B3 tail weight is tiny + B8 seeds avoid it); D2 wires
# it explicitly so the D1 tail's ``end_turn`` cannot flake a KeyError.
# =============================================================================
def _fake_v4_orig_select_fn(name: str):
    import numpy as np

    def _fn(ctx):
        ids = np.asarray(ctx.legal_action_ids, dtype=np.intp)
        return int(ids[0]) if ids.size else 0

    return _fn


def _fake_d_opponent_policies_factory():
    policies: dict = {}
    for name in ("v4-orig-argmax", "v4-orig-t07", "v4-orig-t12"):
        policies[name] = TempV4Opponent(name=name, select_fn=_fake_v4_orig_select_fn(name))
    # self -> a SelfPrevOpponent-equivalent (pick first legal action).
    policies["self"] = _EndTurnPolicy()
    # The 4 POLICY_OPPONENT_KINDS identities (end_turn + greedy_face + self +
    # v4max) -- wire all so any sampled policy identity resolves.
    policies["greedy_face"] = _EndTurnPolicy()
    policies["end_turn"] = _EndTurnPolicy()
    policies["v4max"] = _EndTurnPolicy()
    return policies


# =============================================================================
# Fake runners with a FIXED outcome (for the plateau tests where a controllable
# H2H-vs-best series is required). The B8 ``_FakeGameRunner`` derives the
# outcome from ``seed % mod``; these fix it regardless of seed so the H2H series
# is exactly flat (all-win -> h2h=1.0 at/above target; all-loss -> h2h=0.0
# below target).
# =============================================================================
class _AlwaysWinRunner:
    """Always-win ``BlockBGameRunner`` -> H2H score rate 1.0 (at/above target)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def play(self, opponent_kind: str, *, seed: int, candidate_side: str) -> BlockBGameResult:
        self.calls.append((opponent_kind, int(seed), str(candidate_side)))
        return BlockBGameResult(
            game=GameResult(
                outcome="win",
                mana_draw_count=1,
                eligible_turns=2,
                opponent=str(opponent_kind),
            ),
            candidate_side=str(candidate_side),
        )


class _AlwaysLoseRunner:
    """Always-loss ``BlockBGameRunner`` -> H2H score rate 0.0 (below target)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def play(self, opponent_kind: str, *, seed: int, candidate_side: str) -> BlockBGameResult:
        self.calls.append((opponent_kind, int(seed), str(candidate_side)))
        return BlockBGameResult(
            game=GameResult(
                outcome="loss",
                mana_draw_count=1,
                eligible_turns=2,
                opponent=str(opponent_kind),
            ),
            candidate_side=str(candidate_side),
        )


# =============================================================================
# Shared driver-builder helper. Pre-seeds the pool with a post-C seed anchor
# (D-D2: the caller MUST ``set_seed_anchor(post-C)`` BEFORE ``run()`` so the
# inherited first-snapshot ``set_seed_anchor`` branch is skipped). Defaults:
# mana_draw_baseline=None -> the inherited collapse monitor returns 1.0 always
# (so the mix is the deterministic D-D1 profile, self_share_target=0.50);
# exit_mode="fixed_schedule" (D-D3 default); curriculum_off=True (D-D4 default).
# =============================================================================
def _build_d_driver(
    *,
    n_updates: int = 4,
    snapshot_cadence: int = 2,
    env_count: int = 4,
    steps_per_update: int = 2,
    game_runner=None,
    dominance_target: float = 0.55,
    k_snap: int = DEFAULT_K_SNAP,
    n_snap: int = DEFAULT_BLOCK_B_N_SNAP,
    collapse_boost_factor: float = 2.0,
    mana_draw_baseline: ManaDrawBaseline | None = None,
    self_share_target: float = 0.50,
    exit_mode: str = "fixed_schedule",
    curriculum_off: bool = True,
    e1_candidate_set: E1CandidateSet | None = None,
    pre_seed: bool = True,
    seed_anchor_path: str = "post_c.npz",
    seed: int = 7,
    opponent_policies_factory=None,
) -> d2mod.BlockDLeagueDriver:
    config = _league_config(
        env_count=env_count,
        steps_per_update=steps_per_update,
        snapshot_cadence=snapshot_cadence,
    )
    if game_runner is None:
        # always-win by default -> deterministic h2h=1.0 (at/above target); the
        # plateau tests override with always-lose / always-win explicitly.
        game_runner = _AlwaysWinRunner()
    if opponent_policies_factory is None:
        opponent_policies_factory = _fake_d_opponent_policies_factory
    pool = SnapshotPool(target_non_anchor_count=6)
    if pre_seed:
        # D-D2: fresh pool seeded from post-C (the immutable seed anchor +
        # inaugural best-ever). h2h=0.0 so a real Block-D snapshot at h2h>0
        # strictly beats it (best_ever tracks Block-D improvement).
        pool.set_seed_anchor(
            SnapshotEntry(0, 0.0, seed_anchor_path, 0.0, True)
        )
    return d2mod.BlockDLeagueDriver(
        config,
        pool=pool,
        game_runner=game_runner,
        learner_policy=_FakeLearner(),
        opponent_policies_factory=opponent_policies_factory,
        curriculum=CurriculumReweighter(window_n=3),
        parity=SecondStartParityLoop(window_n=16),
        seed=seed,
        worker_factory=_worker_factory,
        mana_draw_baseline=mana_draw_baseline,
        snapshot_cadence=snapshot_cadence,
        n_snap=n_snap,
        k_snap=k_snap,
        dominance_target=dominance_target,
        collapse_boost_factor=collapse_boost_factor,
        checkpoint_namer=lambda upd: f"fake_ckpt_{upd:04d}.npz",
        games_per_opponent_per_side=1,
        games_per_opponent_gauntlet=1,
        steps_per_update=steps_per_update,
        self_share_target=self_share_target,
        exit_mode=exit_mode,
        curriculum_off=curriculum_off,
        e1_candidate_set=e1_candidate_set,
    )


# Dispatchable identity set (the names A4 ``resolve_opponent_dispatch`` accepts
# after the v5_snapshot->self merge): POLICY_OPPONENT_KINDS |
# BLOCK_B_POLICY_OPPONENT_KINDS | RULE_AGENT_CODES keys. Used by the
# ``_merge_self_snapshot_split`` regression guard.
_DISPATCHABLE = (
    set(rls.POLICY_OPPONENT_KINDS)
    | set(rls.BLOCK_B_POLICY_OPPONENT_KINDS)
    | set(rls.RULE_AGENT_CODES.keys())
)


# =============================================================================
# 1. test_d2_subclass_of_b8
# =============================================================================
class TestSubclassOfB8:
    def test_d2_subclass_of_b8(self):
        driver = _build_d_driver()
        # BlockDLeagueDriver is a SUBCLASS of BlockBLeagueDriver (the inherited
        # _snapshot_step / _measure_snapshot / _collapse_boost_for / _learner_
        # mana_draw_rate come from B8).
        assert isinstance(driver, b8mod.BlockBLeagueDriver)
        assert issubclass(d2mod.BlockDLeagueDriver, b8mod.BlockBLeagueDriver)


# =============================================================================
# 2. test_build_reweighted_mix_uses_d1_not_b3
# =============================================================================
class TestBuildReweightedMixUsesD1:
    def test_build_reweighted_mix_uses_d1_not_b3(self, monkeypatch):
        # The mix handed to A4 is the D1 consolidation profile (self+v5_snapshot
        # share == self_share_target 0.50 EXACTLY, NOT B3's 0.05 cap) with
        # ``v5_snapshot`` MERGED into ``self`` (no v5_snapshot entry reaches A4)
        # and every name dispatchable.
        driver = _build_d_driver(self_share_target=0.50, mana_draw_baseline=None)
        mix = driver._build_reweighted_mix()
        names = [n for n, _ in mix]
        # NO v5_snapshot entry (merged into self).
        assert "v5_snapshot" not in names, (
            "v5_snapshot must be merged into self before the mix reaches A4"
        )
        # The self weight == self_share_target 0.50 EXACTLY (D-D1, not B3's 0.05
        # cap; collapse_boost=1.0 with mana_draw_baseline=None -> no boost).
        self_w = sum(w for n, w in mix if n == "self")
        assert abs(self_w - 0.50) < 1e-9, (
            f"self weight must be self_share_target=0.50, got {self_w}"
        )
        # B3's cap would give self+v5_snapshot = 0.05; D1 gives 0.50 -- the
        # load-bearing consolidation-vs-frozen-field gap.
        assert self_w > 0.05, "D1 self share (0.50) must exceed B3's 0.05 cap"
        # The mix sums to 1.0.
        assert abs(sum(w for _, w in mix) - 1.0) < 1e-9
        # Every name is dispatchable (in POLICY_OPPONENT_KINDS |
        # BLOCK_B_POLICY_OPPONENT_KINDS | RULE_AGENT_CODES, OR "self").
        for n in names:
            assert n in _DISPATCHABLE, f"non-dispatchable identity in mix: {n!r}"

    def test_no_v5_snapshot_reaches_a4(self, monkeypatch):
        # End-to-end: spy on run_live_self_play_update (the D2 module global the
        # copied loop looks up) and assert the opponent_mix_parsed handed to A4
        # has NO v5_snapshot entry + every name dispatchable.
        driver = _build_d_driver(mana_draw_baseline=None)
        real_update = d2mod.run_live_self_play_update
        seen_mixes: list = []

        def spy_update(*args, **kwargs):
            seen_mixes.append(list(kwargs.get("opponent_mix_parsed")))
            return real_update(*args, **kwargs)

        monkeypatch.setattr(d2mod, "run_live_self_play_update", spy_update)
        driver.run(2)
        assert len(seen_mixes) == 2
        for mix in seen_mixes:
            names = [n for n, _ in mix]
            assert "v5_snapshot" not in names
            for n in names:
                assert n in _DISPATCHABLE, f"non-dispatchable identity: {n!r}"


# =============================================================================
# 3. test_curriculum_off_calls_reweight_cap_zero
# =============================================================================
class TestCurriculumOffCapZero:
    def test_curriculum_off_calls_reweight_cap_zero(self, monkeypatch):
        # D-D4 OFF (curriculum_off=True, default): curriculum.reweight is called
        # with cap=0.0 (the NO-OP -- every boost factor 1.0, mix unchanged).
        driver = _build_d_driver(curriculum_off=True, mana_draw_baseline=None)
        real_reweight = driver.curriculum.reweight
        caps: list = []

        def spy_reweight(mix, *, cap=0.25):
            caps.append(float(cap))
            return real_reweight(mix, cap=cap)

        driver.curriculum.reweight = spy_reweight
        driver.run(2)
        assert caps, "curriculum.reweight must be called each update"
        assert all(c == 0.0 for c in caps), (
            f"curriculum_off=True must call reweight with cap=0.0, got {caps}"
        )

    def test_curriculum_on_calls_reweight_cap_0_25(self, monkeypatch):
        # D-D4 ON (curriculum_off=False): curriculum.reweight is called with
        # cap=0.25 (the B4 per-lane-loss reweight).
        driver = _build_d_driver(curriculum_off=False, mana_draw_baseline=None)
        real_reweight = driver.curriculum.reweight
        caps: list = []

        def spy_reweight(mix, *, cap=0.25):
            caps.append(float(cap))
            return real_reweight(mix, cap=cap)

        driver.curriculum.reweight = spy_reweight
        driver.run(2)
        assert caps, "curriculum.reweight must be called each update"
        assert all(c == 0.25 for c in caps), (
            f"curriculum_off=False must call reweight with cap=0.25, got {caps}"
        )


# =============================================================================
# 4. test_fixed_schedule_exit_fires_at_end
# =============================================================================
class TestFixedScheduleExitAtEnd:
    def test_fixed_schedule_exit_fires_at_end(self):
        # D-D3 fixed-schedule: run n_updates=4 with snapshot_cadence=2 -> 2
        # snapshots. Use always-lose + k_snap=1 so B7 plateau FIRES at the last
        # snapshot (h2h=0.0 below target, flat -> exit_fires=True with reason
        # "plateau_below_dominance_target" under the default below_target_exits=
        # True). The D2 run() override IGNORES exit_fires under fixed_schedule:
        # the loop completes + exits with "block_d_schedule_complete".
        driver = _build_d_driver(
            n_updates=4,
            snapshot_cadence=2,
            k_snap=1,
            game_runner=_AlwaysLoseRunner(),
            exit_mode="fixed_schedule",
        )
        manifest = driver.run(4)
        assert manifest.exited_to_e1 is True
        assert manifest.exit_verdict["reason"] == "block_d_schedule_complete"
        # NOT a plateau exit (the D-D3 fixed-schedule path, not the B7 path).
        assert manifest.exit_verdict["reason"] != "plateau_at_or_above_dominance_target"
        assert manifest.exit_verdict["reason"] != "plateau_below_dominance_target"
        # The loop ran to completion (NO early return even though B7 fired).
        assert manifest.n_updates_run == 4
        # B7 DID fire at the last snapshot (exit_fires True) but was ignored --
        # the load-bearing fixed-schedule divergence from B8.
        assert len(manifest.snapshot_history) == 2
        assert manifest.snapshot_history[-1]["exit_fires"] is True, (
            "B7 exit_fires must be True at the last snapshot (always-lose + "
            "k_snap=1 -> plateau below target) to demonstrate it is IGNORED "
            "under fixed_schedule"
        )
        # best_ever_path is set (pre-seeded pool) + candidate_paths non-empty
        # with post-D (best_ever_path) first.
        assert manifest.best_ever_path is not None
        assert manifest.candidate_paths, "candidate_paths must be non-empty"
        assert manifest.candidate_paths[0] == manifest.best_ever_path, (
            "post-D (best_ever_path) must be first in candidate_paths"
        )


# =============================================================================
# 5. test_plateau_mode_early_exit
# =============================================================================
class TestPlateauModeEarlyExit:
    def test_plateau_mode_early_exit(self):
        # D-D3 plateau: exit_mode="plateau" -> below_target_exits=False wired in
        # __init__. Always-win -> h2h=1.0 (at/above target 0.55), flat series.
        # snapshot_cadence=1, n_updates=5, k_snap=3 -> 4 h2h points at update 4
        # -> plateau=True, not below_target -> exit_fires=True with reason
        # "plateau_at_or_above_dominance_target". Early exit at update 4 (< 5).
        driver = _build_d_driver(
            n_updates=5,
            snapshot_cadence=1,
            k_snap=3,
            game_runner=_AlwaysWinRunner(),
            exit_mode="plateau",
        )
        # below_target_exits=False was wired by __init__ for plateau mode.
        assert driver.below_target_exits is False
        manifest = driver.run(5)
        assert manifest.exited_to_e1 is True
        assert manifest.exit_verdict["reason"] == "plateau_at_or_above_dominance_target"
        # NOT the non-firing else-branch reason (that is the DEFAULT
        # below_target_exits=True reading; plateau mode flips it).
        assert manifest.exit_verdict["reason"] != "dominant_plateau_e1_path"
        # Early return: the exit fired at update 4, so n_updates_run < n_updates.
        assert manifest.n_updates_run < 5, (
            "plateau mode must early-return (n_updates_run < n_updates)"
        )
        assert manifest.n_updates_run == 4
        # candidate_paths non-empty with post-D (best_ever_path) first.
        assert manifest.candidate_paths
        assert manifest.candidate_paths[0] == manifest.best_ever_path


# =============================================================================
# 6. test_pre_seed_skips_first_snapshot_seed_anchor
# =============================================================================
class TestPreSeedSkipsFirstSnapshotSeedAnchor:
    def test_pre_seed_skips_first_snapshot_seed_anchor(self):
        # D-D2: the caller pre-sets pool.set_seed_anchor(post-C) BEFORE run();
        # the inherited _snapshot_step:508-509 first-snapshot set_seed_anchor
        # branch is SKIPPED (seed_anchor is already set) -> post-C (NOT the
        # first Block-D snapshot) remains the seed anchor.
        driver = _build_d_driver(
            n_updates=4,
            snapshot_cadence=2,
            pre_seed=True,
            seed_anchor_path="post_c.npz",
        )
        assert driver.pool.seed_anchor is not None
        assert driver.pool.seed_anchor.path == "post_c.npz"
        manifest = driver.run(4)
        # After run: the seed anchor is STILL post-C (the first Block-D snapshot
        # at update 2 was added as a ROLLING non-anchor, not the seed).
        assert driver.pool.seed_anchor.path == "post_c.npz", (
            "pre-seeded post-C must remain the seed anchor (the inherited "
            "first-snapshot set_seed_anchor branch is skipped)"
        )
        # The first Block-D snapshot path is NOT the seed anchor.
        first_snap_path = manifest.snapshot_history[0]["path"]
        assert first_snap_path != "post_c.npz"
        assert driver.pool.seed_anchor.path != first_snap_path


# =============================================================================
# 7. test_uses_b6_not_a_gate
# =============================================================================
class TestUsesB6NotAGate:
    def test_uses_b6_not_a_gate(self, monkeypatch):
        # Regression guard: the snapshot promotion gate is B6
        # ``evaluate_block_b_gate`` (inherited from B8 _snapshot_step), NOT A5
        # ``play_gauntlet`` (play_gauntlet IS used for the gauntlet-rate
        # component inside _measure_snapshot, that is fine; the GUARD is that
        # evaluate_block_b_gate is the promotion gate). Spy on the B8 module
        # global (the inherited _snapshot_step looks up evaluate_block_b_gate in
        # B8's module globals) + assert the snapshot record carries
        # gate_passed/gate_reason.
        driver = _build_d_driver(n_updates=4, snapshot_cadence=2)
        real_eval = b8mod.evaluate_block_b_gate
        eval_calls: list = []

        def spy_eval(*args, **kwargs):
            eval_calls.append(kwargs)
            return real_eval(*args, **kwargs)

        monkeypatch.setattr(b8mod, "evaluate_block_b_gate", spy_eval)
        manifest = driver.run(4)
        # B6 evaluate_block_b_gate is called once per snapshot (the promotion
        # gate).
        assert len(eval_calls) == manifest.n_snapshots == 2, (
            "evaluate_block_b_gate (B6) must be called once per snapshot"
        )
        # Each snapshot record carries the B6 gate verdict fields.
        for snap in manifest.snapshot_history:
            assert "gate_passed" in snap
            assert "gate_reason" in snap
            assert "gate_failed_criteria" in snap


# =============================================================================
# 8. test_d2_loop_matches_b8_per_update_steps
# =============================================================================
class TestD2LoopMatchesB8PerUpdateSteps:
    def test_d2_loop_matches_b8_per_update_steps(self, monkeypatch):
        # Regression guard: the copied run() loop performs the SAME per-update
        # steps as B8 -- build mix -> parity p1/p2 -> run_live_self_play_update
        # with opponent_mix_parsed=mix (the D1 mix, merged, no v5_snapshot) ->
        # curriculum.update -> snapshot cadence -> _snapshot_step. Spy on
        # run_live_self_play_update (the D2 module global the copied loop looks
        # up) and assert it is called once per update with the right args.
        driver = _build_d_driver(
            n_updates=4, snapshot_cadence=2, mana_draw_baseline=None
        )
        real_update = d2mod.run_live_self_play_update
        calls: list = []

        def spy_update(*args, **kwargs):
            calls.append({
                "opponent_mix_parsed": list(kwargs.get("opponent_mix_parsed")),
                "p1_score_rate": kwargs.get("p1_score_rate"),
                "p2_score_rate": kwargs.get("p2_score_rate"),
                "seed": kwargs.get("seed"),
                "steps": kwargs.get("steps"),
            })
            return real_update(*args, **kwargs)

        monkeypatch.setattr(d2mod, "run_live_self_play_update", spy_update)
        manifest = driver.run(4)
        # Called once per update (4 updates -> 4 calls).
        assert len(calls) == 4
        # Each call: opponent_mix_parsed is the D1 mix (merged, no v5_snapshot),
        # seed == base + update_number, p1/p2 from parity.
        for i, c in enumerate(calls):
            update_number = i + 1
            names = [n for n, _ in c["opponent_mix_parsed"]]
            assert "v5_snapshot" not in names, (
                "the mix handed to A4 must have v5_snapshot merged into self"
            )
            # all names dispatchable.
            for n in names:
                assert n in _DISPATCHABLE, f"non-dispatchable identity: {n!r}"
            # seed == driver.seed + update_number (the B8 per-update seeding).
            assert c["seed"] == int(driver.seed) + update_number, (
                f"seed must be base+update_number, call {i}: {c['seed']}"
            )
            # p1/p2 are floats from parity (update 1 is pre-snapshot -> 0.5
            # defaults; this proves the loop reads parity, not a hardcoded
            # value).
            assert isinstance(c["p1_score_rate"], float)
            assert isinstance(c["p2_score_rate"], float)
        # update 1 (pre-snapshot): parity empty -> 0.5/0.5 defaults.
        assert abs(calls[0]["p1_score_rate"] - 0.5) < 1e-9
        assert abs(calls[0]["p2_score_rate"] - 0.5) < 1e-9
        # the manifest records the mix used + the A4 live-path kind per update.
        for m in manifest.update_metrics:
            assert m["update_kind"] == "live_self_play"
            assert m["opponent_mix_parsed"] is True
            assert isinstance(m["mix_used"], list) and m["mix_used"]


# =============================================================================
# 9. test_candidate_paths_thread_e1_set
# =============================================================================
class TestCandidatePathsThreadE1Set:
    def test_candidate_paths_thread_e1_set(self):
        # Pass an E1CandidateSet threaded with post-C3 + post-B; after run()
        # candidate_paths == [best_path, "c3.npz", "b.npz"] (post-D first, then
        # post-C3, then post-B; Nones dropped). The driver attribute reflects
        # the FILLED set (post_d_path == best_path after run).
        e1_set = E1CandidateSet(post_c3_best_path="c3.npz", post_b_path="b.npz")
        driver = _build_d_driver(
            n_updates=4, snapshot_cadence=2, e1_candidate_set=e1_set
        )
        manifest = driver.run(4)
        best_path = driver.pool.best_ever.path
        assert best_path is not None
        assert manifest.candidate_paths == [best_path, "c3.npz", "b.npz"], (
            f"candidate_paths must be [post-D, post-C3, post-B]; got "
            f"{manifest.candidate_paths}"
        )
        # post-D first.
        assert manifest.candidate_paths[0] == best_path
        # The driver attribute reflects the FILLED set (post_d_path set).
        assert driver.e1_candidate_set is not None
        assert driver.e1_candidate_set.post_d_path == best_path
        assert driver.e1_candidate_set.post_c3_best_path == "c3.npz"
        assert driver.e1_candidate_set.post_b_path == "b.npz"


# =============================================================================
# 10. test_candidate_paths_no_e1_set
# =============================================================================
class TestCandidatePathsNoE1Set:
    def test_candidate_paths_no_e1_set(self):
        # e1_candidate_set=None -> candidate_paths == [best_path] (post-D only;
        # no post-C3/post-B threaded).
        driver = _build_d_driver(
            n_updates=4, snapshot_cadence=2, e1_candidate_set=None
        )
        manifest = driver.run(4)
        best_path = driver.pool.best_ever.path
        assert best_path is not None
        assert manifest.candidate_paths == [best_path], (
            f"with no E1 set, candidate_paths == [best_path]; got "
            f"{manifest.candidate_paths}"
        )


# =============================================================================
# 11. test_manifest_to_dict_keys
# =============================================================================
class TestManifestToDictKeys:
    def test_manifest_to_dict_keys(self):
        # BlockDLeagueManifest.to_dict() has exactly the 11 D2 keys (the
        # renamed exited_to_e1 -- NO exited_to_c2 key -- + the 2 new fields
        # candidate_paths + aggregate_history).
        m = d2mod.BlockDLeagueManifest()
        d = m.to_dict()
        expected = {
            "update_metrics",
            "snapshot_history",
            "promotion_decisions",
            "h2h_history",
            "exit_verdict",
            "n_updates_run",
            "n_snapshots",
            "best_ever_path",
            "exited_to_e1",
            "candidate_paths",
            "aggregate_history",
        }
        assert set(d.keys()) == expected, (
            f"to_dict keys mismatch; got {set(d.keys())}"
        )
        # NO exited_to_c2 key (the rename).
        assert "exited_to_c2" not in d
        # fresh-seeded defaults.
        assert d["aggregate_history"] == []
        assert d["candidate_paths"] == []
        assert d["exited_to_e1"] is False
        assert d["n_updates_run"] == 0


# =============================================================================
# 12. test_invalid_exit_mode_raises + test_invalid_self_share_target_raises
# =============================================================================
class TestInvalidArgsRaise:
    def test_invalid_exit_mode_raises(self):
        with pytest.raises(ValueError, match="exit_mode"):
            _build_d_driver(exit_mode="bogus")

    def test_invalid_self_share_target_raises(self):
        with pytest.raises(ValueError, match="self_share_target"):
            _build_d_driver(self_share_target=1.5)
        with pytest.raises(ValueError, match="self_share_target"):
            _build_d_driver(self_share_target=-0.1)

    def test_valid_boundary_self_share_target_accepted(self):
        # 0.0 and 1.0 are the valid boundaries (not raises).
        d0 = _build_d_driver(self_share_target=0.0)
        assert d0.self_share_target == 0.0
        d1 = _build_d_driver(self_share_target=1.0)
        assert d1.self_share_target == 1.0


# =============================================================================
# 13. test_aggregate_history_fresh_seeded
# =============================================================================
class TestAggregateHistoryFreshSeeded:
    def test_aggregate_history_fresh_seeded(self):
        # manifest.aggregate_history starts [] (fresh-seeded at run() entry, NOT
        # carried from any CLoopManifest) and is populated from
        # self._aggregate_history at exit (the B6 monotone-aggregate series
        # maintained by the inherited _snapshot_step).
        fresh = d2mod.BlockDLeagueManifest()
        assert fresh.aggregate_history == [], (
            "aggregate_history must be fresh-seeded [] (NOT carried from "
            "CLoopManifest)"
        )
        driver = _build_d_driver(n_updates=4, snapshot_cadence=2)
        manifest = driver.run(4)
        # At exit, manifest.aggregate_history == list(self._aggregate_history)
        # (the inherited B6 series -- NOT carried from CLoopManifest).
        assert manifest.aggregate_history == list(driver._aggregate_history), (
            "manifest.aggregate_history must be populated from "
            "self._aggregate_history at exit"
        )
        # It is a list of floats (the B6 monotone-aggregate series).
        assert isinstance(manifest.aggregate_history, list)
        for v in manifest.aggregate_history:
            assert isinstance(v, float)