"""Block C component C4 -- ``c_loop_driver.py`` -- the C-loop driver that closes
Block C (NEW). Mirrors B8 ``BlockBLeagueDriver`` STRUCTURE but drives the C2->C3
offline-replay path (NOT A4 live self-play) and uses the D-C6 aggregate
stall-counter as the C->D exit signal (NOT B7 ``detect_h2h_plateau``).

V5-Max pipeline position: Block C components C0-C3 DONE -> this file is C4
(``BLOCK_C_PLAN.md`` section C4 + decisions D-C6 / D-C8).

PURPOSE (``BLOCK_C_PLAN.md:101-104`` + D-C6): per iteration --
  (a) C2 COLLECT: ``collection_driver.collect(mcp_client)`` -> fresh human
      v5_trace group dirs (OR a pre-built ``OfflineReplayBatch``). If group dirs,
      call ``build_offline_replay_batch`` (C2) with the current ``policy_fn`` to
      build the ``OfflineReplayBatch``. A skip (insufficient human data) records
      a skip iteration and continues (NO crash; mirrors C2/C3 skip-gates).
  (b) C3 REPLAY: ``replay.run(offline_replay_batch, checkpoint_path=...,
      save_checkpoint_path=new_path)`` -> ``AwacCrrMetrics`` (MONITORING-ONLY; the
      new checkpoint path is the candidate). A skip (MLX/npz absent) records a
      skip iteration and continues.
  (c) MEASURE (mirrors B8 ``_measure_snapshot``): play the candidate checkpoint
      via ``game_runner`` -> ``h2h_rate`` (vs best-ever), ``gauntlet_rate``
      (A5 ``play_gauntlet``), ``mana_draw_rate``, ``p1_p2_gap`` (B5 side-stratified
      parity). The real path uses ``AsA5GameRunner`` (B8 :177) + A5
      ``play_gauntlet`` + B5 ``play_side_stratified_gauntlet``; tests inject a
      fake ``game_runner`` returning canned rates (the loop is synthetic-testable
      without MLX/Rust/ONNX/rlhf_env DB/socket).
  (d) B6 PROMOTE (NOT A5 a_gate): ``evaluate_block_b_gate`` (B6 :315) with
      ``aggregate_history=self._aggregate_history`` -> ``BlockBGateResult``.
      ``self._aggregate_history`` is FRESH-SEEDED at C-loop entry (an empty ``[]``
      in ``run`` start; NOT inherited from any Block-B history). The gate appends
      the current ``block_b_aggregate`` to ``monotone_aggregate_history``;
      ``self._aggregate_history`` is reassigned to that list (NO double-append).
  (e) B1 SNAPSHOT POOL: ``SnapshotEntry`` (B1 :68) + ``set_seed_anchor`` on the
      FIRST snapshot / ``add_snapshot`` otherwise + ``maybe_update_best_ever``
      (strict H2H improvement; D-C8 B1 best-ever anchor argmax).
  (f) D-C6 STALL-COUNTER (NEW -- the C->D exit signal, NOT B7): on iteration
      >= 2, compare the current ``block_b_aggregate`` vs the PRIOR C3 iteration's
      aggregate; a GAIN = ``current > prior + monotone_tolerance`` (strictly
      increasing beyond FP noise); NOT a gain -> ``stall++``; a gain ->
      ``stall = 0``. ``stall`` does NOT increment on iteration 1 (no prior) NOR on
      a skipped iteration (a skip is not a plateau). DECOUPLED from B6 promote:
      a snapshot can pass B6 promote AND still increment ``stall`` if the
      aggregate did not strictly increase (B6 promote updates best-ever, stall
      counts plateau -- independent signals). If ``stall >= k_stall`` (K=2) ->
      ``exited_to_D=True``, ``exit_verdict``, ``best_ever_path=pool.best_ever.path``,
      break.

Does NOT call B7 ``detect_h2h_plateau`` anywhere (B7 = B->C2 handoff, NOT C->D;
C4 uses the D-C6 aggregate stall-counter instead).

CONSTRAINTS (frozen-classic guard): C4 is a NEW file. NO edit to frozen-classic
/ A1-A5 / B1-B8 / ``block_b_league_driver.py`` / ``block_b_gate.py`` /
``snapshot_pool.py`` / ``a_gate.py`` / ``gauntlet_v5.py`` / ``rust_ppo.py`` /
``rust_collector.py`` / ``bc_train.py`` / ``v5_policy.py`` / ``warm_start_v5.py``
/ core / ``obs_v5.py`` / ``contracts.py`` / ``offline_replay_bridge.py`` (C2) /
``awac_crr_replay.py`` (C3) / ``c2_collection_driver.py`` (C1). All consumed
READ-ONLY. MLX/Rust are NOT imported at module top -- the loop runs via fakes
(mirrors B8 :556-557 "the league loop itself runs without MLX").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# B6 (block_b_gate) + B1 (snapshot_pool) + A5 ManaDrawBaseline -- all MLX/Rust-free
# at module top (a_gate's mlx/rust imports are lazy inside functions :893/:898;
# gauntlet_v5 is pure dataclasses; block_b_gate imports only a_gate; snapshot_pool
# is pure python with a lazy rust_live_self_play import inside one method).
from .a_gate import ManaDrawBaseline
from .block_b_gate import (
    DEFAULT_BLOCK_B_N_SNAP,
    DEFAULT_MONOTONE_TOLERANCE,
    BlockBGateResult,
    block_b_aggregate,
    evaluate_block_b_gate,
)
from .snapshot_pool import SnapshotEntry, SnapshotPool

__all__ = [
    "CLoopDriver",
    "CLoopManifest",
    "CollectionOutcome",
    "MeasurementResult",
    "CollectionDriverProtocol",
    "ReplayProtocol",
    "GameRunnerProtocol",
    "CheckpointNamerProtocol",
    "A5MeasurementRunner",
    "DEFAULT_C_LOOP_K_STALL",
]


#: D-C6 default K=2 -- number of consecutive non-gaining iterations before the
#: C-loop exits to Block D (``BLOCK_C_PLAN.md`` D-C6: "exit→D at stall==K=2").
DEFAULT_C_LOOP_K_STALL: int = 2


# =============================================================================
# Injectable Protocol types (the loop is synthetic-testable with fakes; NO real
# MLX/Rust/ONNX/rlhf_env DB/socket at test time). The real wiring (C1
# C2CollectionDriver + C2 bridge + C3 AwacCrrReplay + A5 AsA5GameRunner + B1
# SnapshotPool) is composed at USER-run time; C4 defines the Protocols + the loop.
# =============================================================================
class CheckpointNamerProtocol(Protocol):
    def __call__(self, update_number: int) -> str: ...


class CollectionDriverProtocol(Protocol):
    """C2 collection surface. ``collect`` returns a ``CollectionOutcome`` (the
    real wiring adapts C1 ``C2CollectionDriver`` -- which returns a
    ``C2CollectionResult`` -- into a ``CollectionOutcome`` carrying the harvested
    group dirs OR a pre-built ``OfflineReplayBatch``)."""

    def collect(self, mcp_client: Any) -> "CollectionOutcome": ...


class ReplayProtocol(Protocol):
    """C3 AWAC/CRR replay surface. ``run`` returns an object with ``.status``
    (``'trained'`` / ``'skipped'``) and ``.new_checkpoint_path`` (the candidate
    checkpoint; the real ``AwacCrrReplay.run`` returns ``AwacCrrMetrics`` which
    carries exactly these fields). MONITORING-ONLY -- NO promote/score field."""

    def run(
        self,
        offline_replay_batch: Any,
        *,
        checkpoint_path: Any,
        save_checkpoint_path: Any = None,
    ) -> Any: ...


class GameRunnerProtocol(Protocol):
    """Measurement surface -- plays the candidate checkpoint on the external
    bench and returns the 4 measured rates. The real path wraps A5
    ``play_gauntlet`` (the gauntlet-rate + mana_draw-rate component) + B5
    ``play_side_stratified_gauntlet`` (the H2H-vs-best + p1/p2-gap component)
    via ``A5MeasurementRunner`` (below); tests inject a fake returning canned
    ``MeasurementResult`` s (no MLX/Rust/ONNX)."""

    def measure(
        self, candidate_checkpoint_path: str, *, seed: int
    ) -> "MeasurementResult": ...


# =============================================================================
# Outcome dataclasses (the loop's collaboration contract)
# =============================================================================
@dataclass
class CollectionOutcome:
    """Result of one C2 collection iteration.

    ``status == 'skipped'`` (insufficient human data / no fresh groups) -> the
    C-loop records a skip iteration and continues (NO crash, NO stall increment).
    ``status == 'ok'`` -> either ``batch`` (a pre-built ``OfflineReplayBatch``)
    OR ``group_dirs`` (v5_trace group dirs the loop feeds to
    ``build_offline_replay_batch`` with the current ``policy_fn``).
    """

    status: str
    group_dirs: list[str] | None = None
    batch: Any = None
    reason: str = ""
    mana_draw_row_count: int = 0


@dataclass
class MeasurementResult:
    """The 4 external-bench rates for one candidate checkpoint (the measure step
    output). ``h2h_rate`` is the H2H-vs-best-ever score rate; ``gauntlet_rate`` is
    the A5 ``play_gauntlet`` aggregate score rate; ``mana_draw_rate`` is the
    candidate's mana_draw usage rate; ``p1_p2_gap`` is the B5 side-stratified
    p1/p2 score-rate gap (``SecondStartParityLoop.gap_for_promotion``)."""

    h2h_rate: float
    gauntlet_rate: float
    mana_draw_rate: float
    p1_p2_gap: float


# =============================================================================
# Manifest (mirrors B8 ``BlockBLeagueManifest`` :203 -- per-iteration metrics +
# snapshot history + promotion decisions + aggregate history + the exit-to-D
# signal + stall count).
# =============================================================================
@dataclass
class CLoopManifest:
    """The C-loop run manifest: per-iteration metrics + per-snapshot promotion
    decisions + the fresh-seeded aggregate history + the D-C6 stall count + the
    exit-to-D signal. Mirrors ``BlockBLeagueManifest`` STRUCTURALLY (B8 :203);
    C4 drives the C2->C3 offline-replay path, so ``iteration_metrics`` carry C2
    collection + C3 replay + measure outputs (not A4 live self-play outputs)."""

    iteration_metrics: list[dict[str, Any]] = field(default_factory=list)
    snapshot_history: list[dict[str, Any]] = field(default_factory=list)
    promotion_decisions: list[dict[str, Any]] = field(default_factory=list)
    aggregate_history: list[float] = field(default_factory=list)
    best_ever_path: str | None = None
    exited_to_D: bool = False
    exit_verdict: dict[str, Any] | str | None = None
    n_iterations_run: int = 0
    stall_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_metrics": list(self.iteration_metrics),
            "snapshot_history": list(self.snapshot_history),
            "promotion_decisions": list(self.promotion_decisions),
            "aggregate_history": [float(x) for x in self.aggregate_history],
            "best_ever_path": self.best_ever_path,
            "exited_to_D": bool(self.exited_to_D),
            "exit_verdict": self.exit_verdict,
            "n_iterations_run": int(self.n_iterations_run),
            "stall_count": int(self.stall_count),
        }


# =============================================================================
# The driver
# =============================================================================
class CLoopDriver:
    """The C-loop driver (composes C2 collect + C3 replay + measure + B6 gate +
    B1 pool + the D-C6 stall-counter).

    ALL collaborators are INJECTABLE (Protocol types): ``collection_driver`` (C1
    ``C2CollectionDriver`` OR a fake), ``replay`` (C3 ``AwacCrrReplay`` OR a
    fake), ``game_runner`` (A5 ``AsA5GameRunner`` via ``A5MeasurementRunner`` OR
    a fake), ``snapshot_pool`` (B1 ``SnapshotPool`` OR a fake),
    ``checkpoint_namer`` (callable ``update_number -> path``). MLX/Rust are NOT
    imported at module top -- the loop runs via fakes (mirrors B8 :556-557).

    Args:
      collection_driver: C2 collection surface (``collect(mcp_client) ->
        CollectionOutcome``).
      replay: C3 AWAC/CRR replay surface (``run(batch, *, checkpoint_path,
        save_checkpoint_path) -> AwacCrrMetrics``-shaped result). MONITORING-ONLY.
      game_runner: measurement surface (``measure(candidate_path, *, seed) ->
        MeasurementResult``).
      snapshot_pool: B1 ``SnapshotPool`` (pool-add + best-ever + seed anchor).
      checkpoint_namer: callable ``(update_number) -> path`` for B1 snapshot paths
        (the C3 candidate checkpoint path; tests use a fake namer, production
        wires the real checkpoint dir).
      mana_draw_baseline: A5 ``ManaDrawBaseline`` B (the Q4 reference rate for the
        B6 mana_draw-band component).
      mcp_client: the MCP client passed to ``collection_driver.collect`` (a fake
        in tests; a real rlhf_env MCP client at USER-run time).
      policy_fn: the CURRENT V5 policy at bridge time
        (``policy_fn(obs_batch, action_features_batch) -> (logits, values,
        mana_draw_logit)``, consumed by ``build_offline_replay_batch`` when the
        collection outcome carries group dirs). None when collection returns a
        pre-built batch (tests).
      initial_checkpoint_path: the checkpoint path used as the C3 replay input on
        the FIRST iteration (before the pool has a best-ever anchor). D-C8: once
        the pool has a best-ever, the replay input is ``pool.best_ever.path``.
      n_snap: D-B1 N_snap=5 (B6 monotone window).
      k_stall: D-C6 K=2 (stall threshold for exit->D).
      monotone_tolerance: per-step tolerance for the D-C6 strict-increase gain
        test (default 0.0 = strict; reused from B6 ``DEFAULT_MONOTONE_TOLERANCE``).
      seed: base RNG seed for the measurement step (passed as ``seed +
        update_number`` so each iteration is seeded distinctly).
      opponent_kinds / games_per_opponent: reserved for the real
        ``A5MeasurementRunner`` wiring (the gauntlet roster + games per opponent);
        unused by the fake-game_runner path.
    """

    def __init__(
        self,
        *,
        collection_driver: CollectionDriverProtocol,
        replay: ReplayProtocol,
        game_runner: GameRunnerProtocol,
        snapshot_pool: SnapshotPool,
        checkpoint_namer: CheckpointNamerProtocol,
        mana_draw_baseline: ManaDrawBaseline | None = None,
        mcp_client: Any = None,
        policy_fn: Callable[..., Any] | None = None,
        initial_checkpoint_path: str | None = None,
        n_snap: int = DEFAULT_BLOCK_B_N_SNAP,
        k_stall: int = DEFAULT_C_LOOP_K_STALL,
        monotone_tolerance: float = DEFAULT_MONOTONE_TOLERANCE,
        seed: int = 0,
        opponent_kinds: tuple[str, ...] | list[str] | None = None,
        games_per_opponent: int = 1,
    ) -> None:
        self.collection_driver = collection_driver
        self.replay = replay
        self.game_runner = game_runner
        self.snapshot_pool = snapshot_pool
        self.checkpoint_namer = checkpoint_namer
        self.mana_draw_baseline = mana_draw_baseline
        self.mcp_client = mcp_client
        self.policy_fn = policy_fn
        self.initial_checkpoint_path = initial_checkpoint_path
        self.n_snap = int(n_snap)
        if self.n_snap <= 0:
            raise ValueError("n_snap must be positive")
        self.k_stall = int(k_stall)
        if self.k_stall <= 0:
            raise ValueError("k_stall must be positive")
        self.monotone_tolerance = float(monotone_tolerance)
        self.seed = int(seed)
        self.opponent_kinds = tuple(opponent_kinds) if opponent_kinds else ()
        self.games_per_opponent = int(games_per_opponent)

        # Fresh-seeded live state (reset at run entry -- NOT inherited from any
        # Block-B history).
        self._aggregate_history: list[float] = []

    # ---- helpers ------------------------------------------------------------
    def _baseline_or_default(self) -> ManaDrawBaseline:
        if self.mana_draw_baseline is not None:
            return self.mana_draw_baseline
        return ManaDrawBaseline(
            mana_draw_count=1, eligible_turns=2, rate=0.5,
            hand_cap=4, mana_draw_base=2, valid=True,
        )

    def _current_checkpoint_path(self) -> str | None:
        """D-C8: the C3 replay input is the B1 best-ever anchor once set; before
        the first snapshot, fall back to ``initial_checkpoint_path``."""
        be = self.snapshot_pool.best_ever
        if be is not None:
            return be.path
        return self.initial_checkpoint_path

    def _build_batch(self, outcome: CollectionOutcome) -> Any:
        """Resolve the ``OfflineReplayBatch`` from a collection outcome. If the
        outcome carries a pre-built batch, return it; otherwise build it from the
        group dirs via the C2 ``build_offline_replay_batch`` (lazy import so the
        module stays MLX/Rust-free at top -- the bridge is non-MLX but the lazy
        import keeps the loop module light and avoids the bc_dataset /
        offline_dataset_loader import chain for fake-only tests)."""
        if outcome.batch is not None:
            return outcome.batch
        # Lazy import -- only hit on the real group-dirs path.
        from .offline_replay_bridge import build_offline_replay_batch

        if self.policy_fn is None:
            raise ValueError(
                "c_loop_driver: collection outcome carries group_dirs but no "
                "policy_fn was injected (required to build the OfflineReplayBatch)"
            )
        return build_offline_replay_batch(
            self.policy_fn, group_dirs=outcome.group_dirs,
        )

    # ---- the per-iteration step ---------------------------------------------
    def _iteration_step(self, update_number: int) -> tuple[dict[str, Any], bool]:
        """Run one C-loop iteration: C2 collect -> C3 replay -> measure -> B6
        gate -> B1 pool -> D-C6 stall update. Returns ``(iter_record, skip)``.

        On a C2/C3 skip, ``skip`` is True and ``iter_record`` carries
        ``status='skipped'`` (no aggregate append, no stall increment, no pool
        mutation). On a full iteration, ``skip`` is False and ``iter_record``
        carries the measured rates + gate verdict + pool decision + the current
        aggregate.
        """
        # (a) C2 COLLECT ------------------------------------------------------
        collection_outcome = self.collection_driver.collect(self.mcp_client)
        if not isinstance(collection_outcome, CollectionOutcome):
            # Defensive: a mis-typed collection result is treated as a skip (the
            # loop NEVER crashes on a collection stub; mirrors C2 skip-gates).
            return (
                {
                    "update_number": int(update_number),
                    "status": "skipped",
                    "skip_reason": "collection_result_not_collection_outcome",
                },
                True,
            )
        if collection_outcome.status == "skipped" or (
            collection_outcome.batch is None and not collection_outcome.group_dirs
        ):
            return (
                {
                    "update_number": int(update_number),
                    "status": "skipped",
                    "skip_reason": collection_outcome.reason or "insufficient",
                    "mana_draw_row_count": int(collection_outcome.mana_draw_row_count),
                },
                True,
            )

        offline_batch = self._build_batch(collection_outcome)

        # (b) C3 REPLAY -------------------------------------------------------
        new_path = str(self.checkpoint_namer(update_number))
        current_checkpoint_path = self._current_checkpoint_path()
        replay_metrics = self.replay.run(
            offline_batch,
            checkpoint_path=current_checkpoint_path,
            save_checkpoint_path=new_path,
        )
        replay_status = getattr(replay_metrics, "status", "trained")
        replay_new_path = getattr(replay_metrics, "new_checkpoint_path", None)
        if replay_status == "skipped":
            return (
                {
                    "update_number": int(update_number),
                    "status": "skipped",
                    "skip_reason": "replay_skipped",
                    "replay_reason": getattr(replay_metrics, "extra", {}).get(
                        "reason", ""
                    ) if hasattr(replay_metrics, "extra") else "",
                },
                True,
            )
        # The candidate checkpoint path is the C3-emitted new path (fall back to
        # the namer path if the replay did not echo it).
        candidate_path = str(replay_new_path) if replay_new_path is not None else new_path

        # (c) MEASURE ---------------------------------------------------------
        measured: MeasurementResult = self.game_runner.measure(
            candidate_path, seed=int(self.seed) + int(update_number),
        )
        h2h_rate = float(measured.h2h_rate)
        gauntlet_rate = float(measured.gauntlet_rate)
        mana_draw_rate = float(measured.mana_draw_rate)
        p1_p2_gap = float(measured.p1_p2_gap)

        # (d) B6 PROMOTE (NOT A5 a_gate) -------------------------------------
        gate_result: BlockBGateResult = evaluate_block_b_gate(
            h2h_rate=h2h_rate,
            gauntlet_rate=gauntlet_rate,
            mana_draw_rate=mana_draw_rate,
            baseline=self._baseline_or_default(),
            p1_p2_gap=p1_p2_gap,
            aggregate_history=self._aggregate_history,
            n_snap=self.n_snap,
            monotone_tolerance=self.monotone_tolerance,
        )
        # The gate appends the current aggregate to ``monotone_aggregate_history``;
        # adopt that list (NO double-append -- the gate owns the append).
        self._aggregate_history = list(gate_result.monotone_aggregate_history)
        current_aggregate = float(self._aggregate_history[-1])

        # (e) B1 SNAPSHOT POOL ------------------------------------------------
        entry = SnapshotEntry(
            update_number=int(update_number),
            h2h_vs_best=h2h_rate,
            path=candidate_path,
            p1_p2_gap=p1_p2_gap,
            promotion_eligible=bool(gate_result.passed),
            role="rolling",
        )
        # Seed anchor on the FIRST snapshot (immutable after first set).
        if gate_result.passed and self.snapshot_pool.seed_anchor is None:
            self.snapshot_pool.set_seed_anchor(entry)
        elif self.snapshot_pool.seed_anchor is not None:
            self.snapshot_pool.add_snapshot(entry)
        # B1 best-ever update (strict H2H improvement; D-C8 B1 best-ever argmax).
        promoted_best_ever = bool(gate_result.passed) and self.snapshot_pool.maybe_update_best_ever(
            entry, h2h_vs_best_score_rate=h2h_rate,
        )

        iter_record = {
            "update_number": int(update_number),
            "status": "ran",
            "candidate_path": candidate_path,
            "collection_mana_draw_rows": int(collection_outcome.mana_draw_row_count),
            "replay_status": replay_status,
            "h2h_rate": h2h_rate,
            "gauntlet_rate": gauntlet_rate,
            "mana_draw_rate": mana_draw_rate,
            "p1_p2_gap": p1_p2_gap,
            "gate_passed": bool(gate_result.passed),
            "gate_reason": str(gate_result.reason),
            "gate_failed_criteria": list(gate_result.failed_criteria()),
            "promoted_best_ever": bool(promoted_best_ever),
            "aggregate": current_aggregate,
        }
        return iter_record, False

    # ---- the main loop ------------------------------------------------------
    def run(self, n_iterations: int) -> CLoopManifest:
        """Run ``n_iterations`` C-loop iterations. Stops early if the D-C6
        stall-counter reaches ``k_stall`` (exit->D carrying
        ``pool.best_ever.path``).

        The C2->C3 path is driven via the injected ``collection_driver`` +
        ``replay`` + ``game_runner`` fakes/real collaborators. MLX/Rust are NOT
        imported -- the loop itself runs without them (mirrors B8 :556-557).
        ``aggregate_history`` is FRESH-SEEDED at run entry (empty ``[]`` -- NOT
        inherited from any Block-B history).
        """
        if int(n_iterations) < 0:
            raise ValueError("n_iterations must be non-negative")
        # Fresh-seed the aggregate history at C-loop entry (NOT inherited).
        self._aggregate_history = []
        manifest = CLoopManifest()
        stall = 0

        for it in range(int(n_iterations)):
            update_number = it + 1
            iter_record, skipped = self._iteration_step(update_number)
            manifest.iteration_metrics.append(iter_record)
            manifest.n_iterations_run = update_number

            if skipped:
                # A skip is not a plateau: NO aggregate append, NO stall
                # increment. Continue to the next iteration (no crash).
                continue

            # (f) D-C6 stall-counter (the C->D exit signal, NOT B7).
            # The gate already appended the current aggregate to
            # ``self._aggregate_history``; use that (NO double-append).
            if update_number >= 2 and len(self._aggregate_history) >= 2:
                current = float(self._aggregate_history[-1])
                prior = float(self._aggregate_history[-2])
                if current > prior + self.monotone_tolerance:
                    stall = 0  # GAIN -> reset.
                else:
                    stall += 1  # flat or decreasing within tolerance -> stall++.
            # iteration 1: stall stays 0 (no prior; NO increment).

            # Record the per-snapshot / promotion decision (mirror B8 :612-621).
            snap_record = {
                "update_number": int(update_number),
                "path": iter_record.get("candidate_path"),
                "h2h_rate": float(iter_record["h2h_rate"]),
                "gauntlet_rate": float(iter_record["gauntlet_rate"]),
                "mana_draw_rate": float(iter_record["mana_draw_rate"]),
                "p1_p2_gap": float(iter_record["p1_p2_gap"]),
                "gate_passed": bool(iter_record["gate_passed"]),
                "gate_reason": str(iter_record["gate_reason"]),
                "promoted_best_ever": bool(iter_record["promoted_best_ever"]),
                "aggregate": float(iter_record["aggregate"]),
                "stall_after": int(stall),
            }
            manifest.snapshot_history.append(snap_record)
            manifest.promotion_decisions.append(
                {
                    "update_number": snap_record["update_number"],
                    "gate_passed": snap_record["gate_passed"],
                    "gate_reason": snap_record["gate_reason"],
                    "promoted_best_ever": snap_record["promoted_best_ever"],
                }
            )
            manifest.stall_count = int(stall)
            if self.snapshot_pool.best_ever is not None:
                manifest.best_ever_path = self.snapshot_pool.best_ever.path

            # D-C6 exit->D: stall reached k_stall -> break carrying best_ever.
            if stall >= self.k_stall:
                manifest.exited_to_D = True
                manifest.exit_verdict = {
                    "reason": "d_c6_aggregate_stall",
                    "stall_count": int(stall),
                    "k_stall": int(self.k_stall),
                    "aggregate_history": list(self._aggregate_history),
                    "best_ever_path": (
                        self.snapshot_pool.best_ever.path
                        if self.snapshot_pool.best_ever is not None
                        else None
                    ),
                }
                manifest.aggregate_history = list(self._aggregate_history)
                manifest.stall_count = int(stall)
                return manifest

        manifest.aggregate_history = list(self._aggregate_history)
        manifest.stall_count = int(stall)
        if self.snapshot_pool.best_ever is not None:
            manifest.best_ever_path = self.snapshot_pool.best_ever.path
        return manifest


# =============================================================================
# A5MeasurementRunner -- the REAL measurement adapter (USER-run wiring). Wraps a
# B5 ``BlockBGameRunner`` (side-stratified) + A5 ``play_gauntlet`` + B5
# ``play_side_stratified_gauntlet`` into the ``GameRunnerProtocol`` ``measure``
# surface. Lazy imports so the module stays MLX/Rust-free at top (the B8
# ``AsA5GameRunner`` + ``second_start_parity`` import chain is only pulled when
# the real adapter is constructed; tests inject a fake ``game_runner`` directly).
# =============================================================================
class A5MeasurementRunner:
    """Real measurement adapter: plays the candidate via A5 ``play_gauntlet``
    (the gauntlet-rate + mana_draw-rate component) + B5
    ``play_side_stratified_gauntlet`` (the H2H-vs-best + p1/p2-gap component),
    mirroring B8 ``_measure_snapshot`` (:427-467). The ``block_b_runner`` is a B5
    ``BlockBGameRunner``; ``AsA5GameRunner`` (B8 :177) adapts it to the A5
    ``GameRunner`` Protocol for ``play_gauntlet``. Lazy imports keep the module
    MLX/Rust-free at top (constructed only on the real path; tests use a fake)."""

    def __init__(
        self,
        block_b_runner: Any,
        *,
        gauntlet_roster: tuple[str, ...] | list[str],
        h2h_opponent_kind: str = "best_ever",
        games_per_opponent_per_side: int = 1,
        games_per_opponent_gauntlet: int = 1,
        parity_loop: Any | None = None,
    ) -> None:
        # Lazy import -- only the real path pulls the B8/B5 chain.
        from .block_b_league_driver import AsA5GameRunner
        from .second_start_parity import (
            SecondStartParityLoop,
            play_side_stratified_gauntlet,
        )

        self._runner = block_b_runner
        self._a5_runner = AsA5GameRunner(block_b_runner, candidate_side="p1")
        self._gauntlet_roster = tuple(gauntlet_roster)
        self._h2h_opponent_kind = str(h2h_opponent_kind)
        self._games_per_opponent_per_side = int(games_per_opponent_per_side)
        self._games_per_opponent_gauntlet = int(games_per_opponent_gauntlet)
        self._parity = parity_loop if parity_loop is not None else SecondStartParityLoop()
        self._play_side_stratified = play_side_stratified_gauntlet

    def measure(self, candidate_checkpoint_path: str, *, seed: int) -> MeasurementResult:
        # Lazy import -- A5 ``play_gauntlet`` (the gauntlet-rate + mana_draw-rate
        # component; NOT a custom gauntlet).
        from .a_gate import play_gauntlet

        side_results = self._play_side_stratified(
            self._runner,
            [self._h2h_opponent_kind],
            games_per_opponent_per_side=self._games_per_opponent_per_side,
            seed=int(seed),
        )
        self._parity.update(side_results)
        h2h_rate = self._score_rate_from_block_b_results(side_results)
        p1_p2_gap = float(self._parity.gap_for_promotion())

        outcomes = play_gauntlet(
            self._a5_runner,
            list(self._gauntlet_roster),
            games_per_opponent=self._games_per_opponent_gauntlet,
            seed=int(seed) + 1,
        )
        gauntlet_rate = float(outcomes.score_rate())
        mana_draw_rate = (
            float(outcomes.mana_draw_rate())
            if int(outcomes.eligible_turns) > 0
            else 0.0
        )
        return MeasurementResult(
            h2h_rate=float(h2h_rate),
            gauntlet_rate=float(gauntlet_rate),
            mana_draw_rate=float(mana_draw_rate),
            p1_p2_gap=float(p1_p2_gap),
        )

    @staticmethod
    def _score_rate_from_block_b_results(results: Any) -> float:
        """Score rate over side-stratified ``BlockBGameResult``s (mirrors B8
        ``_score_rate_from_block_b_results`` :647-667)."""
        if not results:
            return 0.5
        wins = draws = losses = 0
        for r in results:
            outcome = getattr(r.game, "outcome", None)
            if outcome == "win":
                wins += 1
            elif outcome == "draw":
                draws += 1
            else:
                losses += 1
        total = wins + draws + losses
        if total <= 0:
            return 0.5
        return (wins + 0.5 * draws) / float(total)
