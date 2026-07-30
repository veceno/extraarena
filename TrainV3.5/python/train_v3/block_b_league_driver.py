"""Block B component B8 -- ``block_b_league_driver.py`` -- the multi-update live
league driver (NEW). The FINAL Block-B component: composes A4 ``run_live_self_play_
update`` (ONE update) into the league loop with B1-B7.

V5-Max pipeline position: Block A COMPLETE -> Block B; B1-B7 DONE -> this file is
B8 (``BLOCK_B_PLAN.md:640-700``).

PURPOSE (``BLOCK_B_PLAN.md:644-660``): per update --
  (a) build the Block-B mix via B3 ``build_block_b_opponent_mix(pool)`` (D-B5
      hybrid, self-snapshot prevalence grown as the pool fills);
  (b) mana_draw-collapse monitor: if the learner's mana_draw usage drops out of
      the A5 band vs V4-orig lanes, apply B3 ``collapse_reweight_boost(factor)``
      to RAISE the self-snapshot share (the monitor wiring deferred from B3/B5;
      B8 computes the learner mana_draw rate from the last rollout's
      ``mana_draw_taken`` / ``mana_draw_legal`` + the A5 ``check_mana_draw_band``
      baseline, decides the boost factor);
  (c) B4 curriculum reweight: ``CurriculumReweighter.reweight(mix)`` (cap 0.25,
      D-B8) toward lanes the learner is losing to, using per-lane loss from
      ``extract_lane_outcomes`` of the last rollout;
  (d) B5 parity: feed measured p1/p2 score rates (``SecondStartParityLoop``)
      into A4 ``sample_learner_sides`` (via ``run_live_self_play_update``'s
      ``p1_score_rate`` / ``p2_score_rate``);
  (e) call A4 ``run_live_self_play_update`` (Option 1, ``opponent_mix_parsed=
      reweighted_block_b_mix``) -> ONE PPO update; collect the
      ``LiveRolloutBatch`` + metrics;
  (f) B4 ``update(extract_lane_outcomes(rollout))``;
  (g) every ``snapshot_cadence`` updates (D-B4 ~2000): snapshot via B1
      ``add_snapshot`` (on top of ``rust_trainer._save_checkpoint`` :802
      scaffolding) -> run the external-bench gauntlet (``game_runner``) -> B6
      ``evaluate_block_b_gate`` -> B1 ``maybe_update_best_ever`` -> B7
      ``detect_h2h_plateau`` -> emit exit->C2 if the plateau fires.

Continues A's frozen hyperparams (D-B7): entropy=0.01, epochs=6, max_turns=120,
learner_only_reward, decisive_early_end -- all in A3 ``PhaseAPPOConfig``,
unchanged (regression guard ``test_continues_a_hyperparams``).

THE v4-orig-* DISPATCH GAP (deferred from B2/B3, closed by the A4 additive
extension this workflow makes): the Block-B mix (B3) includes
``v4-orig-argmax`` / ``v4-orig-t07`` / ``v4-orig-t12`` (B2 ``TempV4Opponent``
instances). A4's dispatch was V5-canonical ONLY -- ``parse_v5_opponent_mix``
raises on v4-orig-* (not in ``V5_OPPONENT_KINDS``) and
``resolve_opponent_dispatch`` raised (not in ``POLICY_OPPONENT_KINDS``). The
minimal ADDITIVE A4 extension (this workflow, ``rust_live_self_play.py``):
  * ``BLOCK_B_POLICY_OPPONENT_KINDS = frozenset({"v4-orig-argmax",
    "v4-orig-t07","v4-orig-t12"})`` -- a SEPARATE frozenset (does NOT touch
    ``POLICY_OPPONENT_KINDS`` / ``PHASE_A_IDENTITIES``);
  * ``resolve_opponent_dispatch`` additively returns ``(POLICY_DISPATCH, None)``
    for those (check AFTER ``POLICY_OPPONENT_KINDS``, before the raise) -- the
    4-policy / 11-identity Phase-A counts are UNCHANGED;
  * ``run_live_self_play_update(..., opponent_mix_parsed=...)`` optional param --
    if provided, skip ``parse_v5_opponent_mix`` + use the pre-parsed mix
    directly (so v4-orig-* pass through ``sample_opponent_identities``, which
    only samples names+weights, NO parsing).
B8 wires ``opponent_policies`` with B2 ``TempV4Opponent`` instances keyed by
``v4-orig-argmax`` / ``v4-orig-t07`` / ``v4-orig-t12`` (the
``opponent_policies_factory`` callable); A4's
``collect_rust_live_rollout`` :700 calls ``opponent_policies[identity].select(i,
ctx)`` for those. NO Rust edit (``worker.rs`` unchanged -- v4-orig-* are PYTHON
policy opponents, NOT Rust rule codes).

LIVE PATH, NOT trace-pool (``test_uses_live_path_not_trace_pool``): B8 drives A4
``run_live_self_play_update`` / ``collect_rust_live_rollout`` -- it does NOT call
``rust_trainer.train_rust_ppo_trace_files`` replay. The trace-pool loop
scaffolding (``rust_trainer.py:92`` -- metrics list, ``checkpoint_every``,
league manifest) is reused as a STRUCTURAL TEMPLATE only.

Skip-gate when MLX/Rust unbuildable (worktree): the PPO optimizer step is
MLX-gated inside A4 (``run_live_self_play_update`` stops after
``prepare_rust_ppo_batch`` when ``model`` / ``optimizer`` are None, A4
:1004-1006); the league loop itself (collect + curriculum + parity + snapshot +
gate + plateau) is testable WITHOUT MLX via ``FakeWorker`` + a fake
``BlockBGameRunner`` + fake ``opponent_policies``.

CONSTRAINTS (frozen-classic guard, ``BLOCK_B_PLAN.md:860-901``): B8 is a NEW
file. NO edit to ``classic_*/reward_v5/v5_trace/warm_start_v5/run_phase26*/run_v5
_acceptance/league_v5.py/gauntlet_v5.py/opponents_v5.py`` (read-only). A3
``ppo_phaseA_config.py`` read-only (continued hyperparams, D-B7). A5
``a_gate.py`` read-only. B1-B7 read-only (consumed). ``rust_trainer.py``
read-only (``_save_checkpoint`` :802 + ``train_rust_ppo_trace_files`` :92
scaffolding reused, NOT edited). NO Rust edit. NO TrainV3.5-into-prod.
Source-vs-source: A4 ``run_live_self_play_update`` /
``collect_rust_live_rollout`` / ``resolve_opponent_dispatch`` = oracle, B8 = UUT
(composes A4 live + B1-B7). Synthetic tests only (``FakeWorker`` + fake
``BlockBGameRunner`` + fake ``opponent_policies``, no real MLX/Rust/ONNX except
skip-gated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .a_gate import (
    MANA_DRAW_BAND_LOW,
    GameResult,
    ManaDrawBaseline,
    play_gauntlet,
)
from .block_b_gate import (
    DEFAULT_BLOCK_B_N_SNAP,
    BlockBGateResult,
    evaluate_block_b_gate,
)
from .block_b_opponent_mix import (
    build_block_b_opponent_mix,
    collapse_reweight_boost,
)
from .curriculum import CurriculumReweighter, extract_lane_outcomes
from .exit_to_c2 import (
    DEFAULT_BELOW_TARGET_EXITS,
    DEFAULT_K_SNAP,
    DEFAULT_MIN_GAIN,
    ExitToC2Verdict,
    detect_h2h_plateau,
)
from .ppo_phaseA_config import (
    PHASE_A_ENTROPY_COEF,
    PHASE_A_EPOCHS,
    PHASE_A_MAX_TURNS,
    PhaseAPPOConfig,
)
from .rust_live_self_play import (
    BLOCK_B_POLICY_OPPONENT_KINDS,
    PolicyOpponent,
    run_live_self_play_update,
    create_live_self_play_session,
)
from .second_start_parity import (
    BlockBGameResult,
    BlockBGameRunner,
    SecondStartParityLoop,
    play_side_stratified_gauntlet,
)
from .snapshot_pool import SnapshotEntry, SnapshotPool

__all__ = [
    "BlockBLeagueDriver",
    "BlockBLeagueManifest",
    "BLOCK_B_DEFAULT_SNAPSHOT_CADENCE",
    "BLOCK_B_DEFAULT_COLLAPSE_BOOST_FACTOR",
    "AsA5GameRunner",
]


#: D-B4 default snapshot cadence (``BLOCK_B_PLAN.md:161`` ~2000 updates).
BLOCK_B_DEFAULT_SNAPSHOT_CADENCE: int = 2000

#: D-B5 default mana_draw-collapse boost factor applied when the learner's
#: mana_draw usage drops below the A5 band low edge vs V4-orig lanes (raises the
#: self-snapshot share, compressing frozen non-self proportionally via B3
#: ``collapse_reweight_boost``). 1.0 = no boost (in band).
BLOCK_B_DEFAULT_COLLAPSE_BOOST_FACTOR: float = 2.0

#: Default gauntlet roster for the B6 gauntlet-rate component (the Block-B
#: V4-orig + exploit identities; the promotion-bench skill probe). The wired
#: ``game_runner`` plays the candidate vs each; tests inject a fake runner.
_DEFAULT_GAUNTLET_ROSTER: tuple[str, ...] = (
    "v4-orig-argmax",
    "v4-orig-t07",
    "v4-orig-t12",
    "stall",
    "anti_draw_greed",
    "punish_empty_board",
)

#: Default H2H-vs-best opponent kind (the best-ever self-snapshot, played side-
#: stratified for the B6 H2H component + the B5 p1/p2 measurement).
_DEFAULT_H2H_OPPONENT_KIND: str = "best_ever"


# =============================================================================
# A5 GameRunner adapter -- wraps a BlockBGameRunner (side-stratified) into the
# A5 GameRunner Protocol (``a_gate.play_gauntlet`` calls ``.play(opp, *, seed)``
# without a candidate_side). B6 ``measure_gauntlet_rate`` / A5 ``play_gauntlet``
# consume this; the candidate side is fixed (p1) for the gauntlet-rate component
# (the side-stratified H2H run is separate, via ``play_side_stratified_gauntlet``).
# =============================================================================
class AsA5GameRunner:
    """Adapt a ``BlockBGameRunner`` to the A5 ``GameRunner`` Protocol.

    A5 ``play_gauntlet`` (``a_gate.py:753``) calls ``game_runner.play(opp, *,
    seed) -> GameResult`` (no ``candidate_side`` arg). This adapter fixes the
    candidate side (default ``"p1"``) and unwraps the ``BlockBGameResult.game``
    field (the composed A5 ``GameResult``)."""

    def __init__(
        self, block_b_runner: BlockBGameRunner, *, candidate_side: str = "p1"
    ) -> None:
        self._runner = block_b_runner
        self._side = str(candidate_side)

    def play(self, opponent_kind: str, *, seed: int) -> GameResult:
        return self._runner.play(
            opponent_kind, seed=int(seed), candidate_side=self._side
        ).game


# =============================================================================
# Manifest (structural template reuse of rust_trainer.train_rust_ppo_trace_files
# :92 loop scaffolding -- metrics list, snapshot history, promotion decisions,
# exit signal -- but driving the LIVE path via A4, NOT trace-pool replay).
# =============================================================================
@dataclass
class BlockBLeagueManifest:
    """The league-run manifest: per-update metrics + per-snapshot promotion
    decisions + the exit-to-C2 signal. Mirrors ``rust_trainer``'s league
    manifest (``rust_trainer.py:92`` loop scaffolding) STRUCTURALLY; B8 drives
    the live path, so ``update_metrics`` carry A4 ``run_live_self_play_update``
    outputs (not trace-pool replay outputs)."""

    update_metrics: list[dict[str, Any]] = field(default_factory=list)
    snapshot_history: list[dict[str, Any]] = field(default_factory=list)
    promotion_decisions: list[dict[str, Any]] = field(default_factory=list)
    h2h_history: list[float] = field(default_factory=list)
    exit_verdict: dict[str, Any] | None = None
    n_updates_run: int = 0
    n_snapshots: int = 0
    exited_to_c2: bool = False
    best_ever_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_updates_run": int(self.n_updates_run),
            "n_snapshots": int(self.n_snapshots),
            "exited_to_c2": bool(self.exited_to_c2),
            "best_ever_path": self.best_ever_path,
            "update_metrics": list(self.update_metrics),
            "snapshot_history": list(self.snapshot_history),
            "promotion_decisions": list(self.promotion_decisions),
            "h2h_history": list(self.h2h_history),
            "exit_verdict": self.exit_verdict,
        }


# =============================================================================
# The driver
# =============================================================================
class _CheckpointNamer(Protocol):
    def __call__(self, update_number: int) -> str: ...


class BlockBLeagueDriver:
    """The multi-update live league driver (composes A4 ``run_live_self_play_
    update`` + B1-B7).

    Args:
      config: A3 ``PhaseAPPOConfig`` (continued, D-B7 frozen hyperparams
        entropy=0.01 / epochs=6 / max_turns=120 / learner_only_reward /
        decisive_early_end -- UNCHANGED; regression guard
        ``test_continues_a_hyperparams``).
      pool: B1 ``SnapshotPool`` (pool-add + best-ever + seed anchor).
      game_runner: B5 ``BlockBGameRunner`` -- plays one gauntlet game vs
        ``opponent_kind`` from a fixed ``candidate_side`` (the external-bench
        gauntlet, A5 Protocol wired to A4; B6 gauntlet + B5 side-stratified).
      learner_policy: A4 ``LearnerPolicy`` (the live V5 policy; tests use a
        fake).
      opponent_policies_factory: callable ``() -> dict[str, PolicyOpponent]``
        wiring the Block-B policy opponents: B2 ``TempV4Opponent`` for
        ``v4-orig-argmax`` / ``v4-orig-t07`` / ``v4-orig-t12`` + A4
        ``V4MaxOpponent`` + ``SelfPrevOpponent`` + the rule-agent identities.
        The factory is called once at ``run`` start; the dict MUST cover every
        policy-dispatch identity in the Block-B mix (the v4-orig-* + self +
        v5_snapshot + end_turn + greedy_face + v4max). Rule-agent identities
        (stall / anti_draw_greed / punish_empty_board / random / ...) dispatch
        via Rust rule codes -- they do NOT need a policy entry.
      curriculum: B4 ``CurriculumReweighter`` (per-lane-loss reweight, cap 0.25).
      parity: B5 ``SecondStartParityLoop`` (p1/p2 measurement + feedback).
      seed: base RNG seed (passed to A4 ``run_live_self_play_update`` per
        update as ``seed + update_index`` so each update is seeded distinctly).
      worker_factory: ``Callable[[int], Any]`` injected into A4
        ``run_live_self_play_update`` (tests pass a ``FakeWorker`` factory; the
        live path builds a real ``RustBatchWorker`` when None).
      mana_draw_baseline: A5 ``ManaDrawBaseline`` B (the Q4 reference rate for
        the mana_draw-collapse monitor + the B6 band check).
      snapshot_cadence: D-B4 ~2000 updates between snapshots.
      n_snap: D-B1 N_snap=5 (B6 monotone window).
      k_snap: D-B2 K_snap~10 (B7 plateau window).
      dominance_target: D-B3 ~0.55 (B7 below-target exit threshold).
      collapse_boost_factor: the D-B5 mana_draw-collapse boost applied when the
        learner's mana_draw usage drops below the A5 band low edge.
      checkpoint_namer: callable ``(update_number) -> path`` for B1 snapshot
        paths (the ``rust_trainer._save_checkpoint`` :802 scaffolding; tests
        use a fake namer, production wires the real checkpoint dir).
      gauntlet_roster: the B6 gauntlet opponent roster (default the Block-B
        V4-orig + exploit identities).
      h2h_opponent_kind: the H2H-vs-best opponent kind (default ``"best_ever"``
        -- the wired ``game_runner`` plays the candidate vs the best-ever
        snapshot side-stratified).
      games_per_opponent_per_side: B5 side-stratified games per opponent per
        side (H2H-vs-best measurement).
      games_per_opponent_gauntlet: B6 gauntlet games per opponent (the
        gauntlet-rate + mana_draw-rate measurement).
      model / optimizer: optional MLX model + optimizer for the PPO step
        (MLX-gated; None -> A4 stops after ``prepare_rust_ppo_batch``, the
        worktree skip-gate path).
      learning_rate_for_update: optional callable ``(update_number) -> lr`` applied
        to ``optimizer.learning_rate`` before the live PPO update and recorded in metrics.
      steps_per_update: learner transitions per env per update (passed to A4
        ``steps``; overrides ``config.steps_per_update`` when not None).
    """

    def __init__(
        self,
        config: PhaseAPPOConfig,
        *,
        pool: SnapshotPool,
        game_runner: BlockBGameRunner,
        learner_policy: Any,
        opponent_policies_factory: Callable[[], dict[str, PolicyOpponent]],
        curriculum: CurriculumReweighter,
        parity: SecondStartParityLoop,
        seed: int = 0,
        worker_factory: Callable[[int], Any] | None = None,
        mana_draw_baseline: ManaDrawBaseline | None = None,
        snapshot_cadence: int = BLOCK_B_DEFAULT_SNAPSHOT_CADENCE,
        n_snap: int = DEFAULT_BLOCK_B_N_SNAP,
        k_snap: int = DEFAULT_K_SNAP,
        dominance_target: float = 0.55,
        collapse_boost_factor: float = BLOCK_B_DEFAULT_COLLAPSE_BOOST_FACTOR,
        checkpoint_namer: _CheckpointNamer | None = None,
        gauntlet_roster: tuple[str, ...] | list[str] = _DEFAULT_GAUNTLET_ROSTER,
        h2h_opponent_kind: str = _DEFAULT_H2H_OPPONENT_KIND,
        games_per_opponent_per_side: int = 1,
        games_per_opponent_gauntlet: int = 1,
        model: Any = None,
        optimizer: Any = None,
        learning_rate_for_update: Callable[[int], float] | None = None,
        steps_per_update: int | None = None,
        below_target_exits: bool = DEFAULT_BELOW_TARGET_EXITS,
        min_gain: float = DEFAULT_MIN_GAIN,
        opponent_mix_override: list[tuple[str, float]] | None = None,
    ) -> None:
        self.config = config
        self.pool = pool
        self.game_runner = game_runner
        self.learner_policy = learner_policy
        self.opponent_policies_factory = opponent_policies_factory
        self.curriculum = curriculum
        self.parity = parity
        self.seed = int(seed)
        self.worker_factory = worker_factory
        self.mana_draw_baseline = mana_draw_baseline
        self.snapshot_cadence = int(snapshot_cadence)
        if self.snapshot_cadence <= 0:
            raise ValueError("snapshot_cadence must be positive")
        self.n_snap = int(n_snap)
        self.k_snap = int(k_snap)
        self.dominance_target = float(dominance_target)
        self.collapse_boost_factor = float(collapse_boost_factor)
        self.checkpoint_namer = checkpoint_namer or (lambda upd: f"checkpoint_{upd:04d}.npz")
        self.gauntlet_roster = tuple(gauntlet_roster)
        self.h2h_opponent_kind = str(h2h_opponent_kind)
        self.games_per_opponent_per_side = int(games_per_opponent_per_side)
        self.games_per_opponent_gauntlet = int(games_per_opponent_gauntlet)
        self.model = model
        self.optimizer = optimizer
        self.learning_rate_for_update = learning_rate_for_update
        self.steps_per_update = steps_per_update
        self.below_target_exits = bool(below_target_exits)
        self.min_gain = float(min_gain)
        self.opponent_mix_override = (
            None if opponent_mix_override is None else _normalize_mix(opponent_mix_override)
        )

        # Live state.
        self._aggregate_history: list[float] = []
        self._h2h_history: list[float] = []
        self._last_rollout: Any = None
        self._opponent_policies: dict[str, PolicyOpponent] | None = None
        self._rollout_session: Any = None

    # ---- helpers ------------------------------------------------------------
    def _ensure_opponent_policies(self) -> dict[str, PolicyOpponent]:
        if self._opponent_policies is None:
            self._opponent_policies = dict(self.opponent_policies_factory())
        return self._opponent_policies

    def _learner_mana_draw_rate(self) -> float | None:
        """The learner's mana_draw usage from the last rollout's
        ``mana_draw_taken`` / ``mana_draw_legal`` channels (None before the
        first update). ``rate = sum(taken) / sum(legal)`` over learner steps."""
        if self._last_rollout is None:
            return None
        legal = self._last_rollout.mana_draw_legal
        taken = self._last_rollout.mana_draw_taken
        n_legal = int(np_arr_sum(legal))
        if n_legal <= 0:
            return None
        return float(np_arr_sum(taken)) / float(n_legal)

    def _collapse_boost_for(self, mana_draw_rate: float | None) -> float:
        """D-B5 mana_draw-collapse monitor: if the learner's mana_draw usage
        drops BELOW the A5 band low edge (``MANA_DRAW_BAND_LOW * B`` -- the Q5
        blind-lane bias: the learner over-fits "opponent never draws" and stops
        drawing mana itself), apply ``collapse_boost_factor`` to raise the
        self-snapshot share (B3 ``collapse_reweight_boost``). In band or no data
        -> 1.0 (no boost)."""
        if mana_draw_rate is None or self.mana_draw_baseline is None:
            return 1.0
        b = float(self.mana_draw_baseline.rate)
        # Couple the monitor to the A5 ``check_mana_draw_band`` low edge
        # (``a_gate.MANA_DRAW_BAND_LOW`` :103) so a future retune of the frozen
        # A5 band criterion is reflected here, not silently diverged.
        low = float(MANA_DRAW_BAND_LOW) * b
        if mana_draw_rate < low:
            return float(self.collapse_boost_factor)
        return 1.0

    # ---- the per-update mix build (B3 + collapse monitor + B4 curriculum) ---
    def _build_reweighted_mix(self) -> list[tuple[str, float]]:
        """Build the Block-B mix (B3) with the collapse boost (D-B5 monitor) +
        B4 curriculum reweight (cap 0.25, D-B8). Returns a renormalized
        ``[(name, weight)]`` summing to 1.0.

        The B3 self-snapshot split emits BOTH ``self`` (live learner self-play)
        and ``v5_snapshot`` (a prior pool snapshot). For A4 DISPATCH purposes
        both are self-play policy opponents and route through the SAME Python
        policy-opponent loop; ``self`` is in A4 ``POLICY_OPPONENT_KINDS`` while
        ``v5_snapshot`` is NOT (and is intentionally NOT added to
        ``BLOCK_B_POLICY_OPPONENT_KINDS`` -- the plan pins that frozenset to the
        three v4-orig-* identities). B8 merges ``v5_snapshot`` into ``self``
        (summing weights) BEFORE handing the mix to A4, so the dispatchable
        identity set is exactly ``POLICY_OPPONENT_KINDS`` ∪
        ``BLOCK_B_POLICY_OPPONENT_KINDS`` ∪ rule-agent codes. The
        ``opponent_policies_factory`` wires ``self`` -> a ``SelfPrevOpponent``
        (a pool snapshot or the learner argmax) covering BOTH self-play roles.
        """
        md_rate = self._learner_mana_draw_rate()
        boost = self._collapse_boost_for(md_rate)
        if self.opponent_mix_override is None:
            mix = build_block_b_opponent_mix(self.pool, **collapse_reweight_boost(boost))
        else:
            mix = _boost_self_share(self.opponent_mix_override, boost)
        # B4 curriculum reweight toward lanes the learner is losing to.
        reweighted = self.curriculum.reweight(mix, cap=0.25)
        # Merge the self-snapshot split into the dispatchable ``self`` identity.
        return _merge_self_snapshot_split(reweighted)

    # ---- the per-snapshot external-bench measurement ------------------------
    def _measure_snapshot(self, snapshot_seed: int) -> dict[str, Any]:
        """Run the external-bench gauntlet for one snapshot:
          * side-stratified H2H vs best-ever -> H2H score rate + B5 parity
            update (the side-stratified results feed ``SecondStartParityLoop``);
          * A5 gauntlet vs the roster -> gauntlet score rate + mana_draw rate
            (via ``play_gauntlet`` aggregation).

        Returns ``{h2h_rate, gauntlet_rate, mana_draw_rate, p1_p2_gap,
        side_results}``."""
        # Side-stratified H2H vs best-ever (B5 p1/p2 measurement + B6 H2H).
        side_results = play_side_stratified_gauntlet(
            self.game_runner,
            [self.h2h_opponent_kind],
            games_per_opponent_per_side=self.games_per_opponent_per_side,
            seed=snapshot_seed,
        )
        self.parity.update(side_results)
        h2h_rate = _score_rate_from_block_b_results(side_results)
        p1_p2_gap = float(self.parity.gap_for_promotion())

        # A5 gauntlet vs the roster (B6 gauntlet-rate + mana_draw-rate).
        a5_runner = AsA5GameRunner(self.game_runner, candidate_side="p1")
        outcomes = play_gauntlet(
            a5_runner,
            list(self.gauntlet_roster),
            games_per_opponent=self.games_per_opponent_gauntlet,
            seed=snapshot_seed + 1,
        )
        gauntlet_rate = float(outcomes.score_rate())
        mana_draw_rate = (
            float(outcomes.mana_draw_rate())
            if int(outcomes.eligible_turns) > 0
            else 0.0
        )
        return {
            "h2h_rate": float(h2h_rate),
            "gauntlet_rate": float(gauntlet_rate),
            "mana_draw_rate": float(mana_draw_rate),
            "p1_p2_gap": float(p1_p2_gap),
            "side_results": side_results,
        }

    # ---- the snapshot decision step (B1 + B6 + B7) --------------------------
    def _snapshot_step(
        self, update_number: int, snapshot_seed: int
    ) -> dict[str, Any]:
        """Snapshot via B1 pool-add -> B6 promotion gate -> B1 best-ever update
        -> B7 plateau check -> exit->C2 if it fires. Returns the per-snapshot
        record dict."""
        path = str(self.checkpoint_namer(update_number))
        measured = self._measure_snapshot(snapshot_seed)

        # B6 promotion gate (4 external-bench components + monotone aggregate).
        gate_result: BlockBGateResult = evaluate_block_b_gate(
            h2h_rate=measured["h2h_rate"],
            gauntlet_rate=measured["gauntlet_rate"],
            mana_draw_rate=measured["mana_draw_rate"],
            baseline=(
                self.mana_draw_baseline
                if self.mana_draw_baseline is not None
                else ManaDrawBaseline(
                    # Never fabricate a successful Q4 reference. An
                    # unmeasured baseline makes the mana-draw gate fail until
                    # the operational runner supplies field evidence.
                    mana_draw_count=0, eligible_turns=0, rate=0.0,
                    hand_cap=4, mana_draw_base=2, valid=False,
                )
            ),
            p1_p2_gap=measured["p1_p2_gap"],
            aggregate_history=self._aggregate_history,
            n_snap=self.n_snap,
        )
        self._aggregate_history = list(gate_result.monotone_aggregate_history)

        # B1 snapshot entry (pool-add on top of _save_checkpoint scaffolding).
        entry = SnapshotEntry(
            update_number=int(update_number),
            h2h_vs_best=float(measured["h2h_rate"]),
            path=path,
            p1_p2_gap=float(measured["p1_p2_gap"]),
            promotion_eligible=bool(gate_result.passed),
            role="rolling",
        )
        # Seed anchor on the FIRST snapshot (immutable after first set).
        if self.pool.seed_anchor is None and gate_result.passed:
            self.pool.set_seed_anchor(entry)
        elif self.pool.seed_anchor is not None:
            self.pool.add_snapshot(entry)

        # B1 best-ever update (strict H2H improvement only).
        promoted_best = bool(gate_result.passed) and self.pool.maybe_update_best_ever(
            entry, h2h_vs_best_score_rate=float(measured["h2h_rate"])
        )

        # B7 plateau check (H2H-vs-best series).
        self._h2h_history.append(float(measured["h2h_rate"]))
        best_path = (
            self.pool.best_ever.path if self.pool.best_ever is not None else None
        )
        exit_verdict: ExitToC2Verdict = detect_h2h_plateau(
            self._h2h_history,
            dominance_target=self.dominance_target,
            K_snap=self.k_snap,
            min_gain=self.min_gain,
            below_target_exits=self.below_target_exits,
            best_checkpoint_path=best_path,
        )

        return {
            "update_number": int(update_number),
            "path": path,
            "h2h_rate": float(measured["h2h_rate"]),
            "gauntlet_rate": float(measured["gauntlet_rate"]),
            "mana_draw_rate": float(measured["mana_draw_rate"]),
            "p1_p2_gap": float(measured["p1_p2_gap"]),
            "gate_passed": bool(gate_result.passed),
            "gate_reason": str(gate_result.reason),
            "gate_failed_criteria": list(gate_result.failed_criteria()),
            "promoted_best_ever": bool(promoted_best),
            "exit_fires": bool(exit_verdict.exit_fires),
            "exit_reason": str(exit_verdict.reason),
            "aggregate": float(self._aggregate_history[-1]),
            "exit_verdict": exit_verdict.to_dict(),
        }

    # ---- the main loop ------------------------------------------------------
    def run(self, n_updates: int) -> BlockBLeagueManifest:
        """Run ``n_updates`` league updates + snapshot/gate/plateau every
        ``snapshot_cadence`` updates. Stops early if B7 emits exit->C2.

        Uses the LIVE path (A4 ``run_live_self_play_update``), NOT
        ``train_rust_ppo_trace_files`` replay. The PPO optimizer step is
        MLX-gated inside A4 (``model`` / ``optimizer`` None -> stops after
        ``prepare_rust_ppo_batch``); the league loop itself runs without MLX.
        """
        if int(n_updates) < 0:
            raise ValueError("n_updates must be non-negative")
        opponent_policies = self._ensure_opponent_policies()
        manifest = BlockBLeagueManifest()

        for u in range(int(n_updates)):
            update_number = u + 1
            # (a)+(b)+(c) build the reweighted Block-B mix.
            mix = self._build_reweighted_mix()

            # (d) B5 parity feedback: measured p1/p2 rates feed sample_learner_sides.
            p1_rate = float(self.parity.p1_score_rate())
            p2_rate = float(self.parity.p2_score_rate())

            learning_rate: float | None = None
            if self.learning_rate_for_update is not None:
                learning_rate = float(self.learning_rate_for_update(update_number))
                if learning_rate <= 0.0:
                    raise ValueError("learning_rate_for_update must return a positive value")
                if self.optimizer is not None:
                    setattr(self.optimizer, "learning_rate", learning_rate)

            # (e) A4 live update (Option 1: opponent_mix_parsed bypasses
            # parse_v5_opponent_mix so v4-orig-* pass through).
            if self._rollout_session is None:
                self._rollout_session = create_live_self_play_session(
                    self.config,
                    seed=int(self.seed) + update_number,
                    worker_factory=self.worker_factory,
                    p1_score_rate=p1_rate,
                    p2_score_rate=p2_rate,
                    opponent_mix_parsed=mix,
                )
            metrics = run_live_self_play_update(
                self.config,
                self.learner_policy,
                opponent_policies,
                seed=int(self.seed) + update_number,
                model=self.model,
                optimizer=self.optimizer,
                worker_factory=self.worker_factory,
                p1_score_rate=p1_rate,
                p2_score_rate=p2_rate,
                steps=self.steps_per_update,
                opponent_mix_parsed=mix,
                session=self._rollout_session,
            )
            rollout = metrics.get("rollout")
            self._last_rollout = rollout
            # trim the rollout from the persisted metrics (keep it in-memory only
            # -- the manifest records scalar metrics, not the full batch).
            slim_metrics = {
                k: v for k, v in metrics.items() if k not in ("rollout", "ppo_batch")
            }
            slim_metrics["mix_used"] = list(mix)
            slim_metrics["p1_score_rate"] = p1_rate
            slim_metrics["p2_score_rate"] = p2_rate
            if learning_rate is not None:
                slim_metrics["learning_rate"] = learning_rate
            slim_metrics["update_number"] = update_number
            manifest.update_metrics.append(slim_metrics)
            manifest.n_updates_run = update_number

            # (f) B4 curriculum update from the rollout's per-lane outcomes.
            if rollout is not None:
                outcomes = extract_lane_outcomes(rollout)
                self.curriculum.update(outcomes)

            # (g) snapshot cadence -> B1 pool-add + B6 gate + B7 plateau.
            if update_number % self.snapshot_cadence == 0:
                # Rotate only at an explicit league boundary. This lets the new
                # curriculum/snapshot mix bind cleanly without changing an
                # opponent halfway through a battle.
                self._rollout_session.close()
                self._rollout_session = None
                snap_record = self._snapshot_step(
                    update_number, snapshot_seed=int(self.seed) + update_number
                )
                manifest.snapshot_history.append(snap_record)
                manifest.n_snapshots += 1
                manifest.promotion_decisions.append(
                    {
                        "update_number": snap_record["update_number"],
                        "gate_passed": snap_record["gate_passed"],
                        "gate_reason": snap_record["gate_reason"],
                        "promoted_best_ever": snap_record["promoted_best_ever"],
                    }
                )
                if self.pool.best_ever is not None:
                    manifest.best_ever_path = self.pool.best_ever.path
                if snap_record["exit_fires"]:
                    manifest.exited_to_c2 = True
                    manifest.exit_verdict = snap_record["exit_verdict"]
                    manifest.h2h_history = list(self._h2h_history)
                    return manifest

        manifest.h2h_history = list(self._h2h_history)
        if self.pool.best_ever is not None:
            manifest.best_ever_path = self.pool.best_ever.path
        if self._rollout_session is not None:
            self._rollout_session.close()
            self._rollout_session = None
        return manifest


# =============================================================================
# Small helpers (kept module-local so the driver body stays readable)
# =============================================================================
def np_arr_sum(arr: Any) -> int:
    """``int(np.asarray(arr).sum())`` -- sum of a bool/int array (the mana_draw
    channel totals). Imported lazily-via-numpy through the rollout arrays."""
    import numpy as np

    return int(np.asarray(arr).sum())


def _score_rate_from_block_b_results(
    results: list[BlockBGameResult] | tuple[BlockBGameResult, ...]
) -> float:
    """Score rate ``((wins + 0.5*draws) / total)`` over side-stratified
    ``BlockBGameResult``s (the H2H-vs-best measurement). Mirrors A5
    ``compute_score_rate`` (``a_gate.py:131``) but operates on
    ``BlockBGameResult`` (which composes ``GameResult``)."""
    if not results:
        return 0.5
    wins = draws = losses = 0
    for r in results:
        if r.game.outcome == "win":
            wins += 1
        elif r.game.outcome == "draw":
            draws += 1
        else:
            losses += 1
    total = wins + draws + losses
    if total <= 0:
        return 0.5
    return (wins + 0.5 * draws) / float(total)


def _merge_self_snapshot_split(
    mix: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Merge the B3 self-snapshot split (``self`` + ``v5_snapshot``) into the
    single dispatchable ``self`` identity (sum weights), dropping
    ``v5_snapshot``. Other identities are passed through unchanged. The
    returned list preserves order (``self`` keeps its first-seen position) and
    is NOT renormalized (the merge preserves the total weight -- ``self`` +
    ``v5_snapshot`` already sum to the self-snapshot share).

    Rationale (``BLOCK_B_PLAN.md`` §3 B8): A4 ``resolve_opponent_dispatch``
    accepts ``self`` (in ``POLICY_OPPONENT_KINDS``) but NOT ``v5_snapshot``.
    The plan pins ``BLOCK_B_POLICY_OPPONENT_KINDS`` to the three v4-orig-*
    identities, so ``v5_snapshot`` is NOT additively enabled there. Both
    self-play roles route through the SAME Python policy-opponent loop in A4
    ``collect_rust_live_rollout`` (:683-700), so merging into ``self`` (wired
    by ``opponent_policies_factory`` to a ``SelfPrevOpponent``) preserves the
    self-snapshot weight without changing the dispatch identity set.
    """
    merged: dict[str, float] = {}
    order: list[str] = []
    for name, weight in mix:
        if name == "v5_snapshot":
            target = "self"
        else:
            target = name
        if target not in merged:
            merged[target] = 0.0
            order.append(target)
        merged[target] += float(weight)
    return [(name, merged[name]) for name in order]


def _normalize_mix(mix: list[tuple[str, float]]) -> list[tuple[str, float]]:
    rows = [(str(name), float(weight)) for name, weight in mix if float(weight) > 0.0]
    total = sum(weight for _name, weight in rows)
    if total <= 0.0:
        raise ValueError("opponent_mix_override must contain at least one positive weight")
    return [(name, weight / total) for name, weight in rows]


def _boost_self_share(mix: list[tuple[str, float]], boost: float) -> list[tuple[str, float]]:
    boost = float(boost)
    if boost <= 0.0:
        raise ValueError("collapse boost must be positive")
    rows: list[tuple[str, float]] = []
    for name, weight in mix:
        factor = boost if name in {"self", "v5_snapshot"} else 1.0
        rows.append((name, float(weight) * factor))
    return _normalize_mix(rows)
