"""Synthetic tests for the C4 C-loop driver (``c_loop_driver.py``).

ALL collaborators are fakes: FakeCollectionDriver, FakeAwacReplay (canned
AwacCrrMetrics-shaped results + a new checkpoint path), FakeGameRunner (canned
MeasurementResult rates), real SnapshotPool (pure-python, non-MLX), fake
CheckpointNamer. NO real MLX/Rust/ONNX/rlhf_env DB/socket is touched -- the loop
runs via fakes (mirrors B8 :556-557 "the league loop itself runs without MLX").

The tests assert SPECIFIC D-C6 behavior: per-iteration stall counts, exit at the
right iteration, best_ever_path carried, fresh-seed aggregate len, call order,
skip-gates -- NOT trivially-true shape checks.
"""
from __future__ import annotations

import inspect

import pytest

from train_v3.a_gate import ManaDrawBaseline
from train_v3.block_b_gate import DEFAULT_BLOCK_B_N_SNAP
from train_v3.c_loop_driver import (
    DEFAULT_C_LOOP_K_STALL,
    CLoopDriver,
    CLoopManifest,
    CollectionOutcome,
    MeasurementResult,
)
from train_v3.snapshot_pool import SnapshotPool


# =============================================================================
# Fakes
# =============================================================================
class FakeCollectionDriver:
    """Returns a canned sequence of CollectionOutcome s (one per call)."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self._i = 0
        self.call_count = 0

    def collect(self, mcp_client):
        self.call_count += 1
        if self._i < len(self._outcomes):
            o = self._outcomes[self._i]
            self._i += 1
            return o
        # Default: a non-skip empty-batch outcome so the loop can continue.
        return CollectionOutcome(status="ok", batch=object(), reason="default")


class FakeAwacReplay:
    """Returns a canned replay result with .status + .new_checkpoint_path per
    call. Records call order + the checkpoint_path / save_checkpoint_path args."""

    def __init__(self, statuses=None, new_paths=None):
        self._statuses = list(statuses) if statuses else []
        self._new_paths = list(new_paths) if new_paths else []
        self._i = 0
        self.calls = []  # (batch, checkpoint_path, save_checkpoint_path)

    def run(self, offline_replay_batch, *, checkpoint_path, save_checkpoint_path=None):
        self.calls.append((offline_replay_batch, checkpoint_path, save_checkpoint_path))
        if self._i < len(self._statuses):
            status = self._statuses[self._i]
            path = (
                self._new_paths[self._i]
                if self._i < len(self._new_paths)
                else save_checkpoint_path
            )
            self._i += 1
        else:
            status = "trained"
            path = save_checkpoint_path
        return _FakeReplayMetrics(status=status, new_checkpoint_path=path)


class _FakeReplayMetrics:
    """AwacCrrMetrics-shaped (has .status + .new_checkpoint_path + .extra)."""

    def __init__(self, *, status, new_checkpoint_path, reason=""):
        self.status = status
        self.new_checkpoint_path = new_checkpoint_path
        self.extra = {"reason": reason} if reason else {}


class FakeGameRunner:
    """Returns a canned sequence of MeasurementResult s (one per measure call).
    Records call order + the candidate path."""

    def __init__(self, measurements):
        self._measurements = list(measurements)
        self._i = 0
        self.calls = []  # (candidate_path, seed)

    def measure(self, candidate_checkpoint_path, *, seed):
        self.calls.append((candidate_checkpoint_path, seed))
        if self._i < len(self._measurements):
            m = self._measurements[self._i]
            self._i += 1
            return m
        # Default: repeat the last measurement.
        return self._measurements[-1] if self._measurements else MeasurementResult(
            0.5, 0.5, 0.5, 0.0
        )


def _baseline():
    return ManaDrawBaseline(
        mana_draw_count=1, eligible_turns=2, rate=0.5,
        hand_cap=4, mana_draw_base=2, valid=True,
    )


def _ok_collection():
    return CollectionOutcome(status="ok", batch=object(), reason="fake")


def _make_driver(
    *,
    measurements,
    collection_outcomes=None,
    replay_statuses=None,
    n_snap=DEFAULT_BLOCK_B_N_SNAP,
    k_stall=DEFAULT_C_LOOP_K_STALL,
    monotone_tolerance=0.0,
    pool=None,
):
    """Build a CLoopDriver wired with fakes + a real SnapshotPool. Returns
    ``(driver, fake_collection, fake_replay, fake_runner, pool)`` so tests can
    spy on collaborator calls."""
    pool = pool if pool is not None else SnapshotPool()
    coll = FakeCollectionDriver(
        collection_outcomes if collection_outcomes is not None else [_ok_collection()]
    )
    replay = FakeAwacReplay(statuses=replay_statuses)
    runner = FakeGameRunner(measurements)
    namer = _FakeNamer()
    driver = CLoopDriver(
        collection_driver=coll,
        replay=replay,
        game_runner=runner,
        snapshot_pool=pool,
        checkpoint_namer=namer,
        mana_draw_baseline=_baseline(),
        n_snap=n_snap,
        k_stall=k_stall,
        monotone_tolerance=monotone_tolerance,
        initial_checkpoint_path="init.npz",
    )
    return driver, coll, replay, runner, pool


class _FakeNamer:
    def __call__(self, update_number):
        return f"ckpt_{update_number:04d}.npz"


# =============================================================================
# Tests
# =============================================================================
def test_d_c6_stall_increasing_no_exit():
    """D-C6 stall INCREASING: 3 strictly-increasing aggregates -> stall stays 0,
    exited_to_D False, loop runs all 3 iterations."""
    # aggregate = h2h + gauntlet + (1 if md in band) + (1 - gap/0.12).
    # md rate 0.5 in band [0.25,0.75]; gap=0 -> parity term 1.0.
    # iter1: 0.50 + 0.50 + 1 + 1 = 3.00
    # iter2: 0.55 + 0.55 + 1 + 1 = 3.10  (strict gain)
    # iter3: 0.60 + 0.60 + 1 + 1 = 3.20  (strict gain)
    measurements = [
        MeasurementResult(0.50, 0.50, 0.5, 0.0),
        MeasurementResult(0.55, 0.55, 0.5, 0.0),
        MeasurementResult(0.60, 0.60, 0.5, 0.0),
    ]
    driver, coll, replay, runner, pool = _make_driver(measurements=measurements)
    manifest = driver.run(3)
    assert manifest.exited_to_D is False
    assert manifest.stall_count == 0
    assert manifest.n_iterations_run == 3
    assert len(manifest.aggregate_history) == 3
    assert manifest.aggregate_history == [3.0, pytest.approx(3.1), pytest.approx(3.2)]


def test_d_c6_stall_plateau_exit_to_d():
    """D-C6 stall PLATEAU -> exit->D at stall==K=2. aggregates [1.0,1.0,1.0]
    flat -> iter2 stall=1, iter3 stall=2 -> exit. Loop does NOT run a 4th."""
    # Build flat aggregates: vary components but keep sum constant.
    # iter1: h2h=0.50 gauntlet=0.50 md=0.5(in band) gap=0.0 -> 0.5+0.5+1+1 = 3.0
    # iter2: h2h=0.50 gauntlet=0.50 md=0.5 gap=0.0 -> 3.0 (flat)
    # iter3: h2h=0.50 gauntlet=0.50 md=0.5 gap=0.0 -> 3.0 (flat)
    measurements = [
        MeasurementResult(0.50, 0.50, 0.5, 0.0),
        MeasurementResult(0.50, 0.50, 0.5, 0.0),
        MeasurementResult(0.50, 0.50, 0.5, 0.0),
    ]
    driver, coll, replay, runner, pool = _make_driver(measurements=measurements)
    manifest = driver.run(4)
    assert manifest.exited_to_D is True
    assert manifest.stall_count == 2
    assert manifest.n_iterations_run == 3  # did NOT run a 4th iteration
    # best_ever_path carried == pool.best_ever.path
    assert pool.best_ever is not None
    assert manifest.best_ever_path == pool.best_ever.path
    assert len(manifest.aggregate_history) == 3


def test_d_c6_stall_reset_on_gain():
    """D-C6 stall RESET ON GAIN: [1.0, 1.0(stall=1), 1.5(gain->reset 0),
    1.5(stall=1), 1.5(stall=2 -> exit)] -> exit at iteration 5. Assert stall
    reset to 0 at the gain iteration."""
    # Use n_snap=1 so B6 monotone does not block; we only care about the stall.
    # aggregate = h2h + gauntlet + 1 + 1 (md in band, gap=0).
    # iter1: 0.50+0.50 -> 3.0
    # iter2: 0.50+0.50 -> 3.0  (flat -> stall=1)
    # iter3: 0.75+0.75 -> 4.0  (gain -> stall=0)
    # iter4: 0.75+0.75 -> 4.0  (flat -> stall=1)
    # iter5: 0.75+0.75 -> 4.0  (flat -> stall=2 -> exit)
    measurements = [
        MeasurementResult(0.50, 0.50, 0.5, 0.0),
        MeasurementResult(0.50, 0.50, 0.5, 0.0),
        MeasurementResult(0.75, 0.75, 0.5, 0.0),
        MeasurementResult(0.75, 0.75, 0.5, 0.0),
        MeasurementResult(0.75, 0.75, 0.5, 0.0),
    ]
    driver, coll, replay, runner, pool = _make_driver(
        measurements=measurements, n_snap=1,
    )
    manifest = driver.run(6)
    assert manifest.exited_to_D is True
    assert manifest.n_iterations_run == 5
    assert manifest.stall_count == 2
    # Inspect the per-iteration stall_after in snapshot_history.
    stalls = [s["stall_after"] for s in manifest.snapshot_history]
    # iter1 stall=0 (no prior), iter2 stall=1, iter3 stall=0 (reset on gain),
    # iter4 stall=1, iter5 stall=2 (exit).
    assert stalls == [0, 1, 0, 1, 2]


def test_d_c6_iteration1_no_stall():
    """D-C6 iteration-1 NO STALL: a single iteration -> stall=0 (no prior)."""
    measurements = [MeasurementResult(0.50, 0.50, 0.5, 0.0)]
    driver, coll, replay, runner, pool = _make_driver(measurements=measurements)
    manifest = driver.run(1)
    assert manifest.exited_to_D is False
    assert manifest.stall_count == 0
    assert manifest.n_iterations_run == 1
    assert len(manifest.aggregate_history) == 1


def test_d_c6_decoupled_from_b6_promote():
    """D-C6 DECOUPLED from B6 promote: an iteration where B6
    gate_result.passed=True (promote) BUT the aggregate did NOT strictly
    increase -> best_ever updated AND stall++. Assert both (independent)."""
    # n_snap=1 so B6 can pass with a flat 2-element monotone window.
    # iter1: h2h=0.50 gauntlet=0.50 gap=0 -> agg=3.0 ; best_ever seeded at 0.50.
    # iter2: h2h=0.60 (strict H2H improve -> best_ever updates) gauntlet=0.50
    #        gap=0.012 -> parity term 1 - 0.012/0.12 = 1 - 0.1 = 0.9
    #        agg = 0.6 + 0.5 + 1 + 0.9 = 3.0 (FLAT -> not a strict gain -> stall++)
    # All 4 B6 components pass: h2h 0.6>=0.5, gauntlet 0.5>=0.5, md 0.5 in band,
    # gap 0.012<=0.12. n_snap=1 -> monotone over [3.0,3.0] non-decreasing -> pass.
    measurements = [
        MeasurementResult(0.50, 0.50, 0.5, 0.0),
        MeasurementResult(0.60, 0.50, 0.5, 0.012),
    ]
    driver, coll, replay, runner, pool = _make_driver(
        measurements=measurements, n_snap=1, k_stall=3,
    )
    manifest = driver.run(2)
    assert manifest.n_iterations_run == 2
    # B6 promote passed on iter2 AND best_ever was updated AND stall incremented.
    iter2 = manifest.iteration_metrics[1]
    assert iter2["gate_passed"] is True
    assert iter2["promoted_best_ever"] is True
    # stall incremented on iter2 (flat aggregate, decoupled from B6 promote).
    assert manifest.stall_count == 1
    assert manifest.exited_to_D is False  # k_stall=3, only 1 stall.


def test_aggregate_history_fresh_seeded():
    """aggregate_history FRESH-SEEDED: starts empty (NOT inherited from a
    Block-B history); len(aggregate_history) == n_iterations_run (one append per
    iteration, no pre-seeded values)."""
    measurements = [
        MeasurementResult(0.50, 0.50, 0.5, 0.0),
        MeasurementResult(0.55, 0.55, 0.5, 0.0),
        MeasurementResult(0.60, 0.60, 0.5, 0.0),
    ]
    driver, coll, replay, runner, pool = _make_driver(measurements=measurements)
    # Pre-populate the driver's aggregate history to prove run() resets it.
    driver._aggregate_history = [9.9, 8.8, 7.7]
    manifest = driver.run(3)
    assert len(manifest.aggregate_history) == 3
    assert manifest.n_iterations_run == 3
    # No pre-seeded values survived -- the history is the 3 measured aggregates.
    assert 9.9 not in manifest.aggregate_history
    assert 8.8 not in manifest.aggregate_history
    assert 7.7 not in manifest.aggregate_history


def test_b6_promote_not_a5_a_gate():
    """B6 promote (NOT A5 a_gate): the driver calls evaluate_block_b_gate (spy)
    and does NOT call any A5 a_gate promote function. The promotion_decisions
    record gate_passed + gate_reason from BlockBGateResult."""
    measurements = [MeasurementResult(0.50, 0.50, 0.5, 0.0)]
    driver, coll, replay, runner, pool = _make_driver(measurements=measurements)

    # Spy: wrap evaluate_block_b_gate (imported into c_loop_driver namespace).
    import train_v3.c_loop_driver as mod

    called = {"gate": 0}

    def spy(**kwargs):
        called["gate"] += 1
        # Record the kwargs to assert it is the B6 signature, not A5 a_gate.
        called["kwargs"] = dict(kwargs)
        return evaluate_block_b_gate(**kwargs)

    from train_v3.block_b_gate import evaluate_block_b_gate

    original = mod.evaluate_block_b_gate
    mod.evaluate_block_b_gate = spy
    try:
        manifest = driver.run(1)
    finally:
        mod.evaluate_block_b_gate = original

    assert called["gate"] == 1
    # B6 signature carries aggregate_history + n_snap (A5 a_gate does not).
    assert "aggregate_history" in called["kwargs"]
    assert "n_snap" in called["kwargs"]
    assert "p1_p2_gap" in called["kwargs"]
    # promotion_decisions carries gate_passed + gate_reason.
    pd = manifest.promotion_decisions[0]
    assert "gate_passed" in pd
    assert "gate_reason" in pd


def test_no_b7_detect_h2h_plateau():
    """NO B7 detect_h2h_plateau: the driver does NOT import or call
    detect_h2h_plateau. The exit signal is the D-C6 stall-counter, not B7."""
    import train_v3.c_loop_driver as mod

    src = inspect.getsource(mod)
    # The module must not IMPORT the exit_to_c2 module (B7 lives there) nor CALL
    # detect_h2h_plateau. The docstring legitimately MENTIONS B7 by name ("NOT
    # B7 detect_h2h_plateau") -- that mention is not an import or a call.
    assert "from .exit_to_c2" not in src
    assert "import exit_to_c2" not in src
    assert "detect_h2h_plateau(" not in src  # no call site
    assert "ExitToC2" not in src
    # B7 names are not bound in the module namespace (not imported).
    assert "detect_h2h_plateau" not in dir(mod)
    assert "ExitToC2Verdict" not in dir(mod)


def test_composition_call_order():
    """Composition: the driver composes C2 collect -> C3 replay -> measure ->
    B6 gate -> B1 pool in that order. The new checkpoint path from C3 replay is
    the candidate measured + added to the pool."""
    measurements = [MeasurementResult(0.55, 0.55, 0.5, 0.0)]
    driver, coll, replay, runner, pool = _make_driver(measurements=measurements)
    manifest = driver.run(1)

    # C2 collection was called.
    assert coll.call_count == 1
    # C3 replay was called with save_checkpoint_path == namer(1) == ckpt_0001.npz.
    assert len(replay.calls) == 1
    _batch, ckpt_path, save_path = replay.calls[0]
    assert save_path == "ckpt_0001.npz"
    # The first replay input checkpoint is initial_checkpoint_path (no best_ever
    # yet on iteration 1).
    assert ckpt_path == "init.npz"
    # The candidate measured == the C3 new path == save_path.
    measured_path, _seed = runner.calls[0]
    assert measured_path == "ckpt_0001.npz"
    # The pool received that path as the seed anchor (first snapshot).
    assert pool.seed_anchor is not None
    assert pool.seed_anchor.path == "ckpt_0001.npz"
    assert pool.best_ever is not None
    assert pool.best_ever.path == "ckpt_0001.npz"
    assert manifest.best_ever_path == "ckpt_0001.npz"


def test_skip_gates_continue_no_stall_increment():
    """Skip-gates: when collection returns a skip OR replay returns
    status='skipped', the iteration is recorded as skipped and the loop
    CONTINUES (no crash, no stall increment on a skipped iteration)."""
    # iter1: skip (collection insufficient) -- no stall increment.
    # iter2: ok, aggregate A1.
    # iter3: ok, aggregate A1 (flat) -> stall=1.
    # iter4: replay skipped -- no stall increment (stall stays 1).
    # iter5: ok, aggregate A1 (flat) -> stall=2 -> exit.
    measurements = [
        MeasurementResult(0.50, 0.50, 0.5, 0.0),  # iter2
        MeasurementResult(0.50, 0.50, 0.5, 0.0),  # iter3
        MeasurementResult(0.50, 0.50, 0.5, 0.0),  # iter5
    ]
    collection_outcomes = [
        CollectionOutcome(status="skipped", reason="insufficient"),  # iter1
        _ok_collection(),  # iter2
        _ok_collection(),  # iter3
        _ok_collection(),  # iter4 (replay will skip)
        _ok_collection(),  # iter5
    ]
    replay_statuses = ["trained", "trained", "skipped", "trained"]
    driver, coll, replay, runner, pool = _make_driver(
        measurements=measurements,
        collection_outcomes=collection_outcomes,
        replay_statuses=replay_statuses,
        n_snap=1,
    )
    manifest = driver.run(5)
    # iter1 + iter4 are skips; iter2/3/5 ran.
    statuses = [m["status"] for m in manifest.iteration_metrics]
    assert statuses == ["skipped", "ran", "ran", "skipped", "ran"]
    # aggregate_history only has the 3 ran iterations (no append on skips).
    assert len(manifest.aggregate_history) == 3
    # iter5 stall=2 -> exit.
    assert manifest.exited_to_D is True
    assert manifest.stall_count == 2
    assert manifest.n_iterations_run == 5
    # iter4 (skip) did NOT increment stall: the stall went 0 (iter2, no prior
    # since iter1 was a skip and iter2 is update_number=2 but only 1 aggregate
    # so far) -> iter3 stall=1 -> iter4 skip (stall stays 1) -> iter5 stall=2.
    ran_stalls = [s["stall_after"] for s in manifest.snapshot_history]
    # snapshot_history only records ran iterations: iter2, iter3, iter5.
    assert ran_stalls == [0, 1, 2]


def test_c2_collection_skip_only_no_crash():
    """A collection skip on iteration 1 does not crash and produces an empty
    aggregate_history (no append)."""
    collection_outcomes = [CollectionOutcome(status="skipped", reason="insufficient")]
    driver, coll, replay, runner, pool = _make_driver(
        measurements=[MeasurementResult(0.5, 0.5, 0.5, 0.0)],
        collection_outcomes=collection_outcomes,
    )
    manifest = driver.run(1)
    assert manifest.n_iterations_run == 1
    assert manifest.iteration_metrics[0]["status"] == "skipped"
    assert manifest.aggregate_history == []
    assert manifest.exited_to_D is False
    assert manifest.stall_count == 0


def test_cloop_manifest_shape():
    """CLoopManifest shape: to_dict() carries iteration_metrics,
    snapshot_history, promotion_decisions, aggregate_history, best_ever_path,
    exited_to_D, exit_verdict, n_iterations_run, stall_count."""
    measurements = [MeasurementResult(0.50, 0.50, 0.5, 0.0)]
    driver, coll, replay, runner, pool = _make_driver(measurements=measurements)
    manifest = driver.run(1)
    d = manifest.to_dict()
    expected_keys = {
        "iteration_metrics", "snapshot_history", "promotion_decisions",
        "aggregate_history", "best_ever_path", "exited_to_D", "exit_verdict",
        "n_iterations_run", "stall_count",
    }
    assert set(d.keys()) == expected_keys
    # CLoopManifest is the returned type.
    assert isinstance(manifest, CLoopManifest)


def test_no_mlx_rust_import_at_module_top():
    """The loop module does NOT import mlx / rust_ffi at module top (the loop
    runs via fakes). Acceptance: MLX/Rust are NOT imported at module top."""
    import train_v3.c_loop_driver as mod

    src = inspect.getsource(mod)
    # No top-level (module-scope) `import mlx` / `import rust` / `from ... mlx`.
    # Lazy imports inside functions are allowed; check the module does not
    # reference mlx/rust at all (the loop is fake-only).
    assert "import mlx" not in src
    assert "import rust" not in src
    assert "rust_ffi" not in src