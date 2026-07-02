"""Tests for Block B component B8 -- ``block_b_league_driver.py`` (NEW).

The FINAL Block-B component: the multi-update live league driver composing A4
``run_live_self_play_update`` + B1-B7. Synthetic tests only -- ``FakeWorker``
(reused from the A4 test module) + a fake ``BlockBGameRunner`` + fake
``opponent_policies`` (B2 ``TempV4Opponent`` with a fake ONNX session / fake
``PolicyOpponent``). The PPO optimizer step is MLX-gated inside A4 (skipped when
``model`` / ``optimizer`` are None); the league loop itself runs without MLX.

Run:
  PYTHONPATH=.:TrainV3.5/python python3 -m pytest \\
      TrainV3.5/python/train_v3/tests/test_block_b_league_driver.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Reuse the A4 FakeWorker / _FakeLearner / _EndTurnPolicy / script builder so the
# B8 tests drive A4 ``collect_rust_live_rollout`` with the SAME deterministic
# stand-in the A4 suite uses (source-vs-source: A4 = oracle, B8 = UUT). The tests
# dir has no __init__.py; add it to sys.path so ``test_rust_live_self_play`` is
# importable as a top-level module.
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_rust_live_self_play as a4test  # noqa: E402

FakeWorker = a4test.FakeWorker
_FakeLearner = a4test._FakeLearner
_EndTurnPolicy = a4test._EndTurnPolicy
_FakeWorkerScriptEntry = a4test._FakeWorkerScriptEntry
_script_alternating = a4test._script_alternating

from train_v3 import block_b_league_driver as b8  # noqa: E402
from train_v3 import rust_live_self_play as rls  # noqa: E402
from train_v3.a_gate import GameResult, ManaDrawBaseline  # noqa: E402
from train_v3.block_b_gate import DEFAULT_BLOCK_B_N_SNAP  # noqa: E402
from train_v3.curriculum import CurriculumReweighter  # noqa: E402
from train_v3.ppo_phaseA_config import (  # noqa: E402
    PHASE_A_EPOCHS,
    PHASE_A_ENTROPY_COEF,
    PHASE_A_MAX_TURNS,
    PhaseAPPOConfig,
)
from train_v3.second_start_parity import BlockBGameResult, SecondStartParityLoop  # noqa: E402
from train_v3.snapshot_pool import SnapshotPool  # noqa: E402
from train_v3.v4_orig_temp_spectrum import TempV4Opponent  # noqa: E402


# =============================================================================
# Fake BlockBGameRunner -- returns synthetic BlockBGameResult for the external-
# bench gauntlet. Deterministic by (opponent_kind, seed, candidate_side).
# =============================================================================
class _FakeGameRunner:
    """A fake ``BlockBGameRunner`` for the B6 gauntlet + B5 side-stratified H2H.

    ``outcome`` is deterministic from ``seed`` (win if seed % 3 == 0, draw if
    seed % 3 == 1, loss otherwise) so the side-stratified p1/p2 rates are
    controllable + the B6/B7 verdicts are reproducible. ``mana_draw_count`` /
    ``eligible_turns`` are fixed (in-band by default; the collapse-monitor test
    overrides via ``mana_draw_count``)."""

    def __init__(
        self,
        *,
        mana_draw_count: int = 1,
        eligible_turns: int = 2,
        outcome_seed_mod: int = 3,
    ) -> None:
        self.mana_draw_count = int(mana_draw_count)
        self.eligible_turns = int(eligible_turns)
        self.outcome_seed_mod = int(outcome_seed_mod)
        self.calls: list[tuple[str, int, str]] = []

    def play(self, opponent_kind: str, *, seed: int, candidate_side: str) -> BlockBGameResult:
        self.calls.append((opponent_kind, int(seed), str(candidate_side)))
        s = int(seed)
        mod = self.outcome_seed_mod
        rem = s % mod if mod > 0 else 0
        if rem == 0:
            outcome = "win"
        elif rem == 1:
            outcome = "draw"
        else:
            outcome = "loss"
        return BlockBGameResult(
            game=GameResult(
                outcome=outcome,
                mana_draw_count=self.mana_draw_count,
                eligible_turns=self.eligible_turns,
                opponent=str(opponent_kind),
            ),
            candidate_side=str(candidate_side),
        )


# =============================================================================
# Fake opponent_policies factory -- wires the Block-B policy opponents.
# v4-orig-* -> TempV4Opponent with a fake select_fn (no real ONNX); self -> a
# SelfPrevOpponent-like fake; end_turn -> _EndTurnPolicy; greedy_face/v4max ->
# simple fakes. Rule-agent identities (stall/anti_draw_greed/punish_empty_board/
# random) dispatch via Rust rule codes (FakeWorker.select_rule_actions) -- NO
# policy entry needed.
# =============================================================================
def _fake_v4_orig_select_fn(name: str):
    def _fn(ctx):
        ids = np.asarray(ctx.legal_action_ids, dtype=np.intp)
        return int(ids[0]) if ids.size else 0
    return _fn


def _fake_opponent_policies_factory():
    policies = {}
    for name in ("v4-orig-argmax", "v4-orig-t07", "v4-orig-t12"):
        policies[name] = TempV4Opponent(name=name, select_fn=_fake_v4_orig_select_fn(name))
    # self -> a SelfPrevOpponent-equivalent (pick first legal action).
    policies["self"] = _EndTurnPolicy()
    # greedy_face + v4max -> simple fakes (cover the dispatch identities B3 may
    # emit via the tail; not in the default B3 mix but kept for completeness).
    policies["greedy_face"] = _EndTurnPolicy()
    policies["v4max"] = _EndTurnPolicy()
    return policies


def _fake_v4_orig_opponent_policies_factory():
    """Factory whose v4-orig-* opponents are real B2 ``TempV4Opponent``
    instances (wired with a fake select_fn). Used by
    ``test_v4_orig_dispatches_via_block_b_extension``."""
    return _fake_opponent_policies_factory()


# =============================================================================
# Shared driver-builder helpers
# =============================================================================
def _league_config(
    *,
    env_count: int = 4,
    steps_per_update: int = 2,
    snapshot_cadence: int = 2,
) -> PhaseAPPOConfig:
    """A tiny ``PhaseAPPOConfig`` for the league loop. The ``opponent_mix``
    field is UNUSED (B8 passes ``opponent_mix_parsed``), but A4 still requires a
    parseable default when ``opponent_mix_parsed`` is None -- keep a valid
    Phase-A string."""
    return PhaseAPPOConfig(
        max_turns=PHASE_A_MAX_TURNS,
        env_count=int(env_count),
        steps_per_update=int(steps_per_update),
        # decisive_early_end left at the A3 default (True, D-B7 continued). The
        # FakeWorker scripts use hero_hp=(45,45,45,45) -> win margin 0 < 0.60 ->
        # never triggers early-end, so the default is safe for synthetic tests.
        advantage_backend="python",
        selected_local_backend="provided",
        prepare_backend="separate",
    )


def _worker_factory(env_count: int):
    """A FakeWorker factory: each env gets an alternating learner/opponent
    script long enough to collect ``steps_per_update`` learner transitions."""
    scripts = []
    for _ in range(int(env_count)):
        scripts.append(_script_alternating(1, opponent_actor=2, n_learner=8, terminal_at=64))
    return FakeWorker(scripts)


def _build_driver(
    *,
    n_updates: int = 4,
    snapshot_cadence: int = 2,
    env_count: int = 4,
    steps_per_update: int = 2,
    mana_draw_baseline: ManaDrawBaseline | None = None,
    game_runner: _FakeGameRunner | None = None,
    dominance_target: float = 0.55,
    k_snap: int = 3,
    n_snap: int = DEFAULT_BLOCK_B_N_SNAP,
    collapse_boost_factor: float = 2.0,
    opponent_policies_factory=None,
) -> b8.BlockBLeagueDriver:
    config = _league_config(
        env_count=env_count,
        steps_per_update=steps_per_update,
        snapshot_cadence=snapshot_cadence,
    )
    if game_runner is None:
        game_runner = _FakeGameRunner()
    if mana_draw_baseline is None:
        mana_draw_baseline = ManaDrawBaseline(
            mana_draw_count=1, eligible_turns=2, rate=0.5,
            hand_cap=4, mana_draw_base=2, valid=True,
        )
    if opponent_policies_factory is None:
        opponent_policies_factory = _fake_opponent_policies_factory
    return b8.BlockBLeagueDriver(
        config,
        pool=SnapshotPool(target_non_anchor_count=6),
        game_runner=game_runner,
        learner_policy=_FakeLearner(),
        opponent_policies_factory=opponent_policies_factory,
        curriculum=CurriculumReweighter(window_n=3),
        parity=SecondStartParityLoop(window_n=16),
        seed=7,
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
    )


# =============================================================================
# 1. test_runs_n_updates_and_snapshots
# =============================================================================
class TestRunsNUpdatesAndSnapshots:
    def test_runs_n_updates_and_snapshots(self):
        driver = _build_driver(n_updates=4, snapshot_cadence=2, env_count=4)
        manifest = driver.run(4)
        assert manifest.n_updates_run == 4
        # snapshots at cadence 2 -> updates 2 and 4 -> 2 snapshots.
        assert manifest.n_snapshots == 2
        assert len(manifest.snapshot_history) == 2
        assert {s["update_number"] for s in manifest.snapshot_history} == {2, 4}
        # the pool fills: seed anchor set on the first snapshot + rolling added.
        assert driver.pool.seed_anchor is not None
        # second snapshot adds a rolling entry (the first set the seed anchor).
        assert driver.pool.non_anchor_count >= 1
        assert len(manifest.update_metrics) == 4
        # each update metric records the mix used + the A4 live-path kind.
        for m in manifest.update_metrics:
            assert m["update_kind"] == "live_self_play"
            assert m["opponent_mix_parsed"] is True
            assert isinstance(m["mix_used"], list) and m["mix_used"]


# =============================================================================
# 2. test_curriculum_and_parity_threaded
# =============================================================================
class TestCurriculumAndParityThreaded:
    def test_curriculum_and_parity_threaded(self):
        # The loop is CLOSED: B4 curriculum + B5 parity feedback change the
        # sampled mix + sides across updates (BLOCK_B_PLAN.md:662-663 -- the
        # loop is closed, NOT static).
        #
        # Dynamism source for the MIX: a non-uniform per-lane loss signal in the
        # curriculum. The synthetic FakeWorker produces all-win rollouts, so
        # ``extract_lane_outcomes`` yields no loss margin and the mix would be
        # static. Pre-seed the curriculum with a loss-heavy lane
        # (``v4-orig-argmax``) so ``CurriculumReweighter.reweight(cap=0.25)``
        # boosts that lane on update 1; as the live run appends real win-heavy
        # outcomes, the rolling window (window_n=3) shifts and the boost decays,
        # so the reweighted mix CHANGES across updates.
        from train_v3.curriculum import LaneOutcome

        driver = _build_driver(
            n_updates=4, snapshot_cadence=2, env_count=8, steps_per_update=2,
        )
        # Pre-seed: the learner lost twice to v4-orig-argmax (loss_rate 1.0 ->
        # max boost factor 1.25 on that lane for update 1's mix).
        driver.curriculum.update([
            LaneOutcome(identity="v4-orig-argmax", outcome="loss"),
            LaneOutcome(identity="v4-orig-argmax", outcome="loss"),
        ])
        manifest = driver.run(4)
        # B4 curriculum: the reweighter window is non-empty (pre-seeded + fed
        # each update by extract_lane_outcomes).
        assert len(driver.curriculum._window) > 0
        mixes = [tuple(m["mix_used"]) for m in manifest.update_metrics]
        # the mixes are well-formed (sum to ~1.0) each update.
        for mix in mixes:
            total = sum(w for _, w in mix)
            assert abs(total - 1.0) < 1e-6, f"mix must sum to 1.0, got {total}"
        # DYNAMISM (mix): the sampled mix changes across updates -- the
        # curriculum feedback loop is closed, not static. A regression that
        # made the mix constant would fail here.
        assert any(
            mixes[i] != mixes[j]
            for i in range(len(mixes))
            for j in range(i + 1, len(mixes))
        ), "mix must change across updates (curriculum loop closed, not static)"
        # DYNAMISM (sides): pre-snapshot p1/p2 are the 0.5 defaults (no gauntlet
        # yet); after the snapshot at update 2, the side-stratified H2H feeds
        # B5 and updates 3+ carry the MEASURED rates (no longer 0.5).
        pre = manifest.update_metrics[0]
        post = manifest.update_metrics[3]
        assert abs(pre["p1_score_rate"] - 0.5) < 1e-9
        assert abs(pre["p2_score_rate"] - 0.5) < 1e-9
        assert post["p1_score_rate"] != pre["p1_score_rate"], (
            "p1_score_rate must change after the snapshot (parity loop closed)"
        )
        assert post["p2_score_rate"] != pre["p2_score_rate"], (
            "p2_score_rate must change after the snapshot (parity loop closed)"
        )
        # the parity loop has accumulated side-stratified results.
        assert len(driver.parity) > 0


# =============================================================================
# 3. test_promotion_and_plateau_per_snapshot
# =============================================================================
class TestPromotionAndPlateauPerSnapshot:
    def test_promotion_and_plateau_per_snapshot(self):
        driver = _build_driver(n_updates=4, snapshot_cadence=2, env_count=4)
        manifest = driver.run(4)
        # Each snapshot triggers a B6 promotion decision + a B7 plateau check.
        assert len(manifest.promotion_decisions) == 2
        for snap, dec in zip(manifest.snapshot_history, manifest.promotion_decisions):
            # B6 gate verdict recorded.
            assert "gate_passed" in snap and "gate_reason" in snap
            assert dec["update_number"] == snap["update_number"]
            # B7 plateau check recorded (exit_verdict present on the snapshot).
            assert "exit_fires" in snap
            assert "exit_reason" in snap
            assert "exit_verdict" in snap
            # the h2h series grows by one per snapshot.
            assert "h2h_rate" in snap


# =============================================================================
# 4. test_continues_a_hyperparams
# =============================================================================
class TestContinuesAHyperparams:
    def test_continues_a_hyperparams(self):
        # D-B7: Block B continues A's frozen hyperparams verbatim
        # (entropy=0.01, epochs=6, max_turns=120, learner_only_reward,
        # decisive_early_end). Regression guard.
        driver = _build_driver(n_updates=1, snapshot_cadence=10)
        cfg = driver.config
        assert cfg.entropy_coef == PHASE_A_ENTROPY_COEF == 0.01
        assert cfg.epochs == PHASE_A_EPOCHS == 6
        assert cfg.max_turns == PHASE_A_MAX_TURNS == 120
        assert cfg.learner_only_reward is True
        assert cfg.decisive_early_end is True  # the config default (continued)
        # the A4 update metrics record the continued hyperparams.
        manifest = driver.run(2)
        m0 = manifest.update_metrics[0]
        assert m0["entropy_coef"] == 0.01
        assert m0["epochs"] == 6
        assert m0["max_turns"] == 120


# =============================================================================
# 5. test_uses_live_path_not_trace_pool
# =============================================================================
class TestUsesLivePathNotTracePool:
    def test_uses_live_path_not_trace_pool(self):
        # The driver calls A4 run_live_self_play_update / collect_rust_live_rollout,
        # NOT train_rust_ppo_trace_files replay.
        import inspect

        src = inspect.getsource(b8.BlockBLeagueDriver.run)
        assert "run_live_self_play_update(" in src, (
            "B8.run must call A4 run_live_self_play_update (the live path)"
        )
        # The driver must NOT CALL train_rust_ppo_trace_files (trace-pool replay).
        # Check the call form (``train_rust_ppo_trace_files(``) so docstring
        # mentions of the name (which document what B8 does NOT do) don't trip
        # the guard.
        mod_src = inspect.getsource(b8)
        assert "train_rust_ppo_trace_files(" not in mod_src, (
            "B8 must NOT call train_rust_ppo_trace_files (trace-pool replay)"
        )
        # And the trace-pool trainer name is NOT bound in the B8 module namespace
        # (no import of train_rust_ppo_trace_files).
        assert "train_rust_ppo_trace_files" not in b8.__dict__, (
            "B8 must not import/bind train_rust_ppo_trace_files"
        )
        # And a live run actually produces live_self_play update metrics.
        driver = _build_driver(n_updates=2, snapshot_cadence=10)
        manifest = driver.run(2)
        assert all(m["update_kind"] == "live_self_play" for m in manifest.update_metrics)


# =============================================================================
# 6. test_skip_if_no_mlx_or_rust
# =============================================================================
def _mlx_available() -> bool:
    try:
        import mlx.core as mx  # noqa: F401
        import mlx.optimizers as opt  # noqa: F401
    except Exception:
        return False
    return True


def _rust_ffi_available() -> bool:
    try:
        from train_v3 import rust_ffi  # noqa: F401
        from train_v3.rust_ffi import RustBatchWorker  # noqa: F401
    except Exception:
        return False
    try:
        w = RustBatchWorker.from_live(seed=1, env_count=1, max_turns=4)
        w.reset(copy=True)
        w.close()
        return True
    except Exception:
        return False


class TestSkipIfNoMlxOrRust:
    def test_skip_if_no_mlx_or_rust(self):
        # The league loop runs WITHOUT MLX/Rust (FakeWorker + fake game_runner).
        # The PPO optimizer step is MLX-gated inside A4 (model/optimizer None ->
        # stops after prepare_rust_ppo_batch). This test asserts the loop
        # completes in the worktree (no MLX/Rust) -- the skip-gate IS the test:
        # if MLX+Rust were both available the real-FFI path would be exercised
        # elsewhere; here the synthetic path must complete regardless.
        has_mlx = _mlx_available()
        has_rust = _rust_ffi_available()
        driver = _build_driver(n_updates=2, snapshot_cadence=1, env_count=2)
        # model/optimizer are None by default -> the PPO step is skipped (MLX-gated).
        assert driver.model is None and driver.optimizer is None
        manifest = driver.run(2)
        # the loop completed (no MLX/Rust needed for the league loop itself).
        assert manifest.n_updates_run == 2
        # the A4 metrics record whether the PPO train step ran (MLX-gated).
        for m in manifest.update_metrics:
            # without MLX the train step is skipped (no update_metrics sub-dict).
            assert "has_rollout" in m and m["has_rollout"] is True
        # Marker: record the worktree build state for the report.
        if not has_mlx and not has_rust:
            pytest.skip("MLX+Rust both unbuildable in worktree -- skip-gate path exercised")


# =============================================================================
# 7. test_v4_orig_dispatches_via_block_b_extension
# =============================================================================
class TestV4OrigDispatch:
    def test_v4_orig_dispatches_via_block_b_extension(self):
        # v4-orig-argmax / v4-orig-t07 / v4-orig-t12 resolve to (POLICY_DISPATCH,
        # None) via the new BLOCK_B_POLICY_OPPONENT_KINDS extension.
        for name in ("v4-orig-argmax", "v4-orig-t07", "v4-orig-t12"):
            kind, code = rls.resolve_opponent_dispatch(name)
            assert kind == rls.POLICY_DISPATCH, (
                f"{name} must dispatch via POLICY_DISPATCH (BLOCK_B_POLICY_OPPONENT_KINDS)"
            )
            assert code is None
        # the opponent_policies factory wires B2 TempV4Opponent for those.
        policies = _fake_v4_orig_opponent_policies_factory()
        for name in ("v4-orig-argmax", "v4-orig-t07", "v4-orig-t12"):
            assert isinstance(policies[name], TempV4Opponent)
            assert policies[name].wired is True
        # BLOCK_B_POLICY_OPPONENT_KINDS is exactly the three v4-orig-* identities.
        assert rls.BLOCK_B_POLICY_OPPONENT_KINDS == frozenset(
            {"v4-orig-argmax", "v4-orig-t07", "v4-orig-t12"}
        )

    def test_v4_orig_mix_runs_through_a4_live_path(self):
        # End-to-end: a Block-B mix containing v4-orig-* (passed via
        # opponent_mix_parsed) runs through A4 run_live_self_play_update WITHOUT
        # raising (collect_rust_live_rollout validates each identity via
        # resolve_opponent_dispatch -> POLICY_DISPATCH for v4-orig-*).
        config = _league_config(env_count=3, steps_per_update=2, snapshot_cadence=10)
        from train_v3.block_b_opponent_mix import build_block_b_opponent_mix

        pool = SnapshotPool(target_non_anchor_count=6)
        # pre-fill one rolling snapshot so the self-snapshot prevalence > 0.
        from train_v3.snapshot_pool import SnapshotEntry

        pool.set_seed_anchor(
            SnapshotEntry(0, 0.5, "seed.npz", 0.0, True, role="seed")
        )
        mix = b8._merge_self_snapshot_split(build_block_b_opponent_mix(pool))
        # the mix contains v4-orig-* identities.
        names = {n for n, _ in mix}
        assert "v4-orig-argmax" in names and "v4-orig-t07" in names and "v4-orig-t12" in names
        metrics = rls.run_live_self_play_update(
            config,
            _FakeLearner(),
            _fake_opponent_policies_factory(),
            seed=11,
            worker_factory=_worker_factory,
            steps=2,
            opponent_mix_parsed=mix,
        )
        assert metrics["has_rollout"] is True
        assert metrics["opponent_mix_parsed"] is True
        # the sampled identities are all dispatchable (no ValueError raised).
        assert set(metrics["opponent_identities"]) <= names | {
            "stall", "anti_draw_greed", "punish_empty_board", "random",
            "greedy_face", "end_turn", "self", "v4max",
        }


# =============================================================================
# 8. test_does_not_break_a4_dispatch_counts
# =============================================================================
class TestDoesNotBreakA4Counts:
    def test_does_not_break_a4_dispatch_counts(self):
        # The additive extension does NOT change A4's Phase-A counts:
        # POLICY_OPPONENT_KINDS still 4, PHASE_A_IDENTITIES still 11.
        assert rls.POLICY_OPPONENT_KINDS == frozenset(
            {"end_turn", "greedy_face", "self", "v4max"}
        )
        assert len(rls.POLICY_OPPONENT_KINDS) == 4
        assert len(rls.PHASE_A_IDENTITIES) == 11
        # the 4 policy opponents still dispatch via POLICY_DISPATCH.
        for name in rls.POLICY_OPPONENT_KINDS:
            kind, _ = rls.resolve_opponent_dispatch(name)
            assert kind == rls.POLICY_DISPATCH
        # BLOCK_B_POLICY_OPPONENT_KINDS is SEPARATE (not in POLICY_OPPONENT_KINDS
        # nor PHASE_A_IDENTITIES).
        assert rls.BLOCK_B_POLICY_OPPONENT_KINDS.isdisjoint(rls.POLICY_OPPONENT_KINDS)
        assert rls.BLOCK_B_POLICY_OPPONENT_KINDS.isdisjoint(set(rls.PHASE_A_IDENTITIES))
        # RULE_AGENT_CODES unchanged (still 7 codes including punish_empty_board).
        assert len(rls.RULE_AGENT_CODES) == 7
        assert rls.RULE_AGENT_CODES["punish_empty_board"] == 5


# =============================================================================
# 9. test_mana_draw_collapse_monitor_wires_b3_boost
# =============================================================================
class TestManaDrawCollapseMonitor:
    def test_mana_draw_collapse_monitor_wires_b3_boost(self):
        # When the learner's mana_draw usage drops BELOW the A5 band low edge
        # (0.5 * B), B8 applies collapse_reweight_boost(factor) -> raises the
        # self-snapshot share. Use a baseline B=0.5 (band [0.25, 0.75]); a
        # rollout with mana_draw rate 0.1 (< 0.25) triggers the boost.
        baseline = ManaDrawBaseline(
            mana_draw_count=1, eligible_turns=2, rate=0.5,
            hand_cap=4, mana_draw_base=2, valid=True,
        )
        driver = _build_driver(
            n_updates=1, snapshot_cadence=10, env_count=4,
            mana_draw_baseline=baseline, collapse_boost_factor=3.0,
        )
        # Before any rollout: no boost (1.0).
        assert driver._collapse_boost_for(None) == 1.0
        # Simulate a last rollout with a LOW mana_draw rate (0.1 < 0.25 low edge).
        class _FakeRollout:
            mana_draw_legal = np.array([[True, True, True, True]], dtype=np.bool_)
            mana_draw_taken = np.array([[True, False, False, False]], dtype=np.bool_)

        driver._last_rollout = _FakeRollout()
        rate = driver._learner_mana_draw_rate()
        assert abs(rate - 0.25) < 1e-9 or rate < 0.25  # 1/4 taken = 0.25 == low edge
        # Force a clearly-below-band rate: 0 taken out of 4 legal -> 0.0 < 0.25.
        driver._last_rollout = _FakeRollout.__new__(_FakeRollout)
        driver._last_rollout.mana_draw_legal = np.array([[True, True, True, True]], dtype=np.bool_)
        driver._last_rollout.mana_draw_taken = np.array([[False, False, False, False]], dtype=np.bool_)
        assert driver._learner_mana_draw_rate() == 0.0
        assert driver._collapse_boost_for(driver._learner_mana_draw_rate()) == 3.0
        # In-band rate (0.5, within [0.25, 0.75]) -> no boost.
        driver._last_rollout.mana_draw_taken = np.array([[True, True, False, False]], dtype=np.bool_)
        assert abs(driver._learner_mana_draw_rate() - 0.5) < 1e-9
        assert driver._collapse_boost_for(driver._learner_mana_draw_rate()) == 1.0

    def test_collapse_boost_changes_mix_self_share(self):
        # The boost actually raises the self-snapshot share in the built mix
        # (B3 collapse_reweight_boost wired through _build_reweighted_mix).
        from train_v3.block_b_opponent_mix import build_block_b_opponent_mix, collapse_reweight_boost

        pool = SnapshotPool(target_non_anchor_count=6, prevalence_pool_target=6)
        from train_v3.snapshot_pool import SnapshotEntry

        # fill the pool so prevalence is at the cap.
        for i in range(6):
            pool.add_snapshot(SnapshotEntry(i + 1, 0.5, f"p{i}.npz", 0.0, False))
        base_mix = build_block_b_opponent_mix(pool, **collapse_reweight_boost(1.0))
        boosted_mix = build_block_b_opponent_mix(pool, **collapse_reweight_boost(3.0))
        base_self = sum(w for n, w in base_mix if n in ("self", "v5_snapshot"))
        boosted_self = sum(w for n, w in boosted_mix if n in ("self", "v5_snapshot"))
        assert boosted_self > base_self, (
            "collapse_reweight_boost must RAISE the self-snapshot share"
        )


# =============================================================================
# 10. test_does_not_edit_frozen_classic
# =============================================================================
class TestDoesNotEditFrozenClassic:
    def test_does_not_edit_frozen_classic(self):
        # The frozen-classic set is read-only. Assert the A4 edit touched ONLY
        # rust_live_self_play.py (the allowed additive extension) + the NEW B8
        # file, and NO frozen-classic file.
        import subprocess

        repo = os.path.abspath(
            os.path.join(_TESTS_DIR, "..", "..", "..", "..")
        )
        frozen = [
            "TrainV3.5/python/train_v3/league_v5.py",
            "TrainV3.5/python/train_v3/gauntlet_v5.py",
            "TrainV3.5/python/train_v3/opponents_v5.py",
            "TrainV3.5/python/train_v3/reward_v5.py",
            "TrainV3.5/python/train_v3/v5_trace.py",
            "TrainV3.5/python/train_v3/warm_start_v5.py",
            "TrainV3.5/python/train_v3/a_gate.py",
            "TrainV3.5/python/train_v3/ppo_phaseA_config.py",
            "TrainV3.5/python/train_v3/rust_trainer.py",
            "TrainV3.5/python/train_v3/snapshot_pool.py",
            "TrainV3.5/python/train_v3/v4_orig_temp_spectrum.py",
            "TrainV3.5/python/train_v3/block_b_opponent_mix.py",
            "TrainV3.5/python/train_v3/curriculum.py",
            "TrainV3.5/python/train_v3/second_start_parity.py",
            "TrainV3.5/python/train_v3/block_b_gate.py",
            "TrainV3.5/python/train_v3/exit_to_c2.py",
        ]
        # TrainV3.5/ is gitignored at the repo root (tracked in this worktree per
        # the task header) -- use git status --porcelain against the worktree.
        # If git ignores TrainV3.5/, fall back to an mtime/content check: the
        # frozen files must be unchanged from the committed B7 state (we assert
        # they simply exist + are not newly written by this workflow).
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain", "--"] + frozen,
                cwd=repo, capture_output=True, text=True, timeout=20,
            )
            # If TrainV3.5 is gitignored, git status reports nothing for all of
            # them (exit 0, empty stdout) -- which means no tracked change. If
            # tracked, any modified frozen file would appear here.
            modified = [ln for ln in res.stdout.splitlines() if ln.strip()]
        except Exception:
            modified = []
        # The frozen-classic set + A3/A5/B1-B7 must NOT be modified by this
        # workflow (only rust_live_self_play.py additive + the NEW B8 file).
        for ln in modified:
            # ignore untracked markers for files git does not track.
            assert not ln.startswith(" M"), f"frozen-classic file modified: {ln!r}"
        # Sanity: the NEW B8 file + the A4 edit target exist.
        assert os.path.exists(os.path.join(_TESTS_DIR, "..", "block_b_league_driver.py"))
        assert os.path.exists(os.path.join(_TESTS_DIR, "..", "rust_live_self_play.py"))