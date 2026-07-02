"""Block D component D2 -- ``block_d_league_driver.py`` -- the Block-D league
driver (NEW). The post-C CONSOLIDATION multi-update driver: a SUBCLASS of B8
``BlockBLeagueDriver`` that overrides TWO surfaces (``_build_reweighted_mix`` +
``run()``), composes A4 ``run_live_self_play_update`` + B1/B5/B6/B7 the same way
B8 does, and emits the D->E1 handoff (``exited_to_e1`` + ``candidate_paths``).

V5-Max pipeline position: Block A + B + C in-worktree COMPLETE -> Block D (D1
DONE, D3 DONE, this file is D2). D2 sits BETWEEN D1 (the consolidation opponent
mix) and E1 (the tournament + ship block): D1 builds the self/peer-heavy mix
(D-D1 ~0.50), D3 builds the fresh post-C-seeded pool + threads the E1 candidate
set, D2 DRIVES the short self-play league (the per-update loop) and at exit
fills the post-D candidate on the ``E1CandidateSet``.

WHY A SUBCLASS WITH TWO OVERRIDES (``BLOCK_D_PLAN.md:65-74``): the inherited B8
``run()`` (``block_b_league_driver.py:550-633``) CANNOT be used verbatim -- it
constructs a ``BlockBLeagueManifest`` at ``:562`` (a local var a subclass cannot
redirect), early-returns on B7 ``exit_fires`` setting ``exited_to_c2`` at
``:624-628`` (the Block-B -> C2 handoff), and at schedule end sets only
``h2h_history`` + ``best_ever_path`` (``:630-633``). D2 needs instead:
  * a ``BlockDLeagueManifest`` (``exited_to_e1`` rename + ``candidate_paths`` +
    ``aggregate_history`` fresh-seeded);
  * under D-D3 fixed-schedule (default): MONITOR the B7 plateau (it still runs
    inside the inherited ``_snapshot_step``) but do NOT early-return on
    ``exit_fires`` -- exit to E1 at schedule end carrying ``best_ever_path``;
  * under D-D3 plateau: a dominant plateau AT/ABOVE the dominance target fires
    ``reason='plateau_at_or_above_dominance_target'`` (B7 with
    ``below_target_exits=False``, wired at ``__init__``) -> early D->E1 exit.

The ``_build_reweighted_mix`` override swaps B3 ``build_block_b_opponent_mix``
(the Block-B 0.05-cap weak-learner profile) for D1 ``build_block_d_opponent_mix``
(the D-D1 ~0.50 consolidation profile) and applies the D-D4 no-op curriculum
(``cap=0.0``) when ``curriculum_off=True``. The ``_merge_self_snapshot_split``
helper (IMPORTED from B8, not redefined) merges ``v5_snapshot`` into ``self`` so
the mix is dispatchable by A4 (``v5_snapshot`` is absent from
``POLICY_OPPONENT_KINDS`` / ``BLOCK_B_POLICY_OPPONENT_KINDS`` /
``RULE_AGENT_CODES``).

THE COPIED PER-UPDATE LOOP (``BLOCK_D_PLAN.md:5`` + D2 spec): B8 has NO factored
``_run_update`` helper -- the loop is inline at ``block_b_league_driver.py:
564-628`` -- and ``block_b_league_driver.py`` is NOT editable (frozen-classic
guard, ``BLOCK_D_PLAN.md:93``), so D2 DUPLICATES the inline loop body and keeps
it in sync with B8. Divergence risk noted; the regression guard
``test_d2_loop_matches_b8_per_update_steps`` pins the per-update step sequence
(build mix -> B5 parity p1/p2 -> A4 ``run_live_self_play_update`` with
``opponent_mix_parsed=mix`` -> ``curriculum.update`` -> snapshot cadence ->
inherited ``_snapshot_step``). Only ``_snapshot_step`` (``:470-547``) and
``_measure_snapshot`` (``:427-467``) are genuinely inherited (called, not
copied); ``_merge_self_snapshot_split`` is IMPORTED (not redefined).

CONSTRAINTS (frozen-classic guard, ``BLOCK_D_PLAN.md:88-94``): D2 is a NEW
subclass module. NO edit to ``block_b_league_driver.py`` / ``block_b_opponent
_mix.py`` / ``classic_*`` / ``reward_v5`` / ``v5_trace`` / ``core`` / ``state``
/ ``league_v5`` / ``gauntlet_v5`` / ``opponents_v5`` / ``rust_ffi`` / ``rust_ppo``
/ ``rust_live_self_play`` / ``c_to_d_handoff`` / ``block_d_opponent_mix`` (all
READ-ONLY). ``_merge_self_snapshot_split`` is IMPORTED from
``block_b_league_driver`` (not redefined). NO Rust edit. NO TrainV3.5-into-prod.
Source-vs-source: B8 ``BlockBLeagueDriver`` = oracle (subclass READ-ONLY
inheritance), D1 ``build_block_d_opponent_mix`` + D3 ``E1CandidateSet`` =
collaborators (DONE), D2 ``BlockDLeagueDriver`` = UUT. Synthetic tests only
(``FakeWorker`` + fake ``BlockBGameRunner`` + fake ``opponent_policies``, no
real MLX/Rust/ONNX except the MLX-gated PPO step which is skipped when
``model`` / ``optimizer`` are None).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .a_gate import ManaDrawBaseline
from .block_b_gate import DEFAULT_BLOCK_B_N_SNAP
from .block_b_league_driver import (
    BLOCK_B_DEFAULT_COLLAPSE_BOOST_FACTOR,
    BLOCK_B_DEFAULT_SNAPSHOT_CADENCE,
    BlockBLeagueDriver,
    _DEFAULT_GAUNTLET_ROSTER,
    _merge_self_snapshot_split,
)
from .block_d_opponent_mix import build_block_d_opponent_mix
from .c_to_d_handoff import E1CandidateSet
from .curriculum import CurriculumReweighter, extract_lane_outcomes
from .exit_to_c2 import (
    DEFAULT_BELOW_TARGET_EXITS,
    DEFAULT_K_SNAP,
    DEFAULT_MIN_GAIN,
)
from .ppo_phaseA_config import PhaseAPPOConfig
from .rust_live_self_play import PolicyOpponent, run_live_self_play_update
from .second_start_parity import BlockBGameRunner, SecondStartParityLoop
from .snapshot_pool import SnapshotPool

__all__ = [
    "BlockDLeagueDriver",
    "BlockDLeagueManifest",
]


# =============================================================================
# Manifest -- a FRESH dataclass mirroring ``BlockBLeagueManifest`` (B8 :202-231)
# fields + the ``exited_to_c2`` -> ``exited_to_e1`` rename + 2 NEW fields
# (``candidate_paths`` + ``aggregate_history``). NOT inherited from
# ``BlockBLeagueManifest``: the field rename + new fields make a fresh dataclass
# cleaner (the plan says the inherited fields are UNCHANGED in VALUE-semantics,
# not by Python inheritance -- ``BLOCK_D_PLAN.md:70``).
# =============================================================================
@dataclass
class BlockDLeagueManifest:
    """The Block-D league-run manifest: per-update metrics + per-snapshot
    promotion decisions + the D->E1 exit signal + the E1 candidate set.

    Mirrors ``BlockBLeagueManifest`` (``block_b_league_driver.py:202-231``)
    STRUCTURALLY with the ``exited_to_c2`` -> ``exited_to_e1`` RENAME (Block D
    exits to E1, not C2 -- ``BLOCK_D_PLAN.md:107``) and TWO new fields:
      * ``candidate_paths`` -- the E1 tournament set (``design.md:134``): post-D
        (``best_ever_path``, filled by D2 at exit) first, then post-C3 best, then
        post-B fallback (Nones dropped); threaded from the D3
        ``E1CandidateSet``.
      * ``aggregate_history`` -- the B6 monotone-aggregate series, FRESH-SEEDED
        ``[]`` at ``run()`` entry (NOT carried from ``CLoopManifest`` -- C's
        aggregate is the C3-replay bench context and mixing it into Block D's B6
        monotone window would context-mix; ``BLOCK_D_PLAN.md:70``).
    """

    update_metrics: list[dict[str, Any]] = field(default_factory=list)
    snapshot_history: list[dict[str, Any]] = field(default_factory=list)
    promotion_decisions: list[dict[str, Any]] = field(default_factory=list)
    h2h_history: list[float] = field(default_factory=list)
    exit_verdict: dict[str, Any] | None = None
    n_updates_run: int = 0
    n_snapshots: int = 0
    best_ever_path: str | None = None
    # RENAMED from ``BlockBLeagueManifest.exited_to_c2`` (Block D -> E1, not C2).
    exited_to_e1: bool = False
    # NEW -- the E1 candidate checkpoint paths (post-D first, then post-C3, then
    # post-B; Nones dropped). Filled by ``run()`` at exit from the
    # ``E1CandidateSet`` (D3).
    candidate_paths: list[str] = field(default_factory=list)
    # NEW -- the B6 monotone-aggregate series, fresh-seeded ``[]`` at ``run()``
    # entry (NOT carried from ``CLoopManifest``).
    aggregate_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_updates_run": int(self.n_updates_run),
            "n_snapshots": int(self.n_snapshots),
            "exited_to_e1": bool(self.exited_to_e1),
            "best_ever_path": self.best_ever_path,
            "update_metrics": list(self.update_metrics),
            "snapshot_history": list(self.snapshot_history),
            "promotion_decisions": list(self.promotion_decisions),
            "h2h_history": list(self.h2h_history),
            "exit_verdict": self.exit_verdict,
            "candidate_paths": list(self.candidate_paths),
            "aggregate_history": list(self.aggregate_history),
        }


# =============================================================================
# The driver -- a SUBCLASS of B8 ``BlockBLeagueDriver`` overriding TWO surfaces.
# =============================================================================
class BlockDLeagueDriver(BlockBLeagueDriver):
    """The Block-D post-C consolidation league driver (D2).

    SUBCLASS of B8 ``BlockBLeagueDriver`` (``block_b_league_driver.py:241``)
    overriding TWO surfaces:
      * ``_build_reweighted_mix`` -- swaps B3 for D1 (D-D1 ~0.50 consolidation
        profile) + the D-D4 no-op curriculum (``cap=0.0`` when
        ``curriculum_off``) + ``_merge_self_snapshot_split`` (IMPORTED) so
        ``v5_snapshot`` is merged into ``self`` before the mix reaches A4.
      * ``run`` -- builds a ``BlockDLeagueManifest`` (NOT
        ``BlockBLeagueManifest``), COPIES the B8 per-update loop body (B8 has no
        factored ``_run_update`` helper; ``block_b_league_driver.py:564-628``
        is duplicated + kept in sync -- divergence risk noted, regression guard
        ``test_d2_loop_matches_b8_per_update_steps``), and emits the D->E1
        handoff (``exited_to_e1`` + ``candidate_paths``). Under D-D3
        fixed-schedule (default) the B7 plateau is MONITORED but its
        ``exit_fires`` is IGNORED (no early return; exit at schedule end). Under
        D-D3 plateau a dominant plateau fires the early D->E1 exit.

    The inherited ``_snapshot_step`` / ``_measure_snapshot`` (B1+B6+B7) are
    called verbatim (NOT copied); ``_merge_self_snapshot_split`` is IMPORTED
    (NOT redefined).

    Args (inherited from B8 ``__init__`` :299-327, passed through to
    ``super().__init__``): ``config``, ``pool``, ``game_runner``,
    ``learner_policy``, ``opponent_policies_factory``, ``curriculum``,
    ``parity``, ``seed``, ``worker_factory``, ``mana_draw_baseline``,
    ``snapshot_cadence``, ``n_snap``, ``k_snap``, ``dominance_target``,
    ``collapse_boost_factor``, ``checkpoint_namer``, ``gauntlet_roster``,
    ``h2h_opponent_kind``, ``games_per_opponent_per_side``,
    ``games_per_opponent_gauntlet``, ``model``, ``optimizer``,
    ``steps_per_update``, ``min_gain``.

    NEW kwargs (D2, at the END):
      self_share_target: D-D1 consolidation self+v5_snapshot share (default
        0.50). Applied DIRECTLY to D1 ``build_block_d_opponent_mix`` (NOT via
        ``pool.self_snapshot_prevalence_weight()`` -- Block D's pool is
        pre-seeded by D3).
      exit_mode: D-D3 exit condition -- ``"fixed_schedule"`` (default; run
        ``n_updates``, B6-promote at cadence, exit->E1 at schedule end) or
        ``"plateau"`` (a dominant B7 plateau fires the early D->E1 exit). When
        ``"plateau"``, ``below_target_exits=False`` is passed to
        ``super().__init__`` so the inherited ``_snapshot_step:528``
        ``detect_h2h_plateau`` uses the flipped at/above reading (reason
        ``"plateau_at_or_above_dominance_target"``).
      curriculum_off: D-D4 -- when True (default) the B4 per-lane-loss reweight
        is a NO-OP (``cap=0.0``); when False the B4 ``cap=0.25`` reweight runs.
      e1_candidate_set: the D3 ``E1CandidateSet`` (post-C3 + post-B threaded;
        post-D None until D2 fills it at exit). May be None (-> ``candidate_paths
        = [best_ever_path]``).
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
        checkpoint_namer: Callable[[int], str] | None = None,
        gauntlet_roster: tuple[str, ...] | list[str] = _DEFAULT_GAUNTLET_ROSTER,
        h2h_opponent_kind: str = "best_ever",
        games_per_opponent_per_side: int = 1,
        games_per_opponent_gauntlet: int = 1,
        model: Any = None,
        optimizer: Any = None,
        steps_per_update: int | None = None,
        min_gain: float = DEFAULT_MIN_GAIN,
        self_share_target: float = 0.50,
        exit_mode: str = "fixed_schedule",
        curriculum_off: bool = True,
        e1_candidate_set: E1CandidateSet | None = None,
    ) -> None:
        if exit_mode not in ("fixed_schedule", "plateau"):
            raise ValueError(
                "exit_mode must be 'fixed_schedule' or 'plateau', got "
                f"{exit_mode!r}"
            )
        if not (0.0 <= float(self_share_target) <= 1.0):
            raise ValueError(
                f"self_share_target must be in [0.0, 1.0], "
                f"got {self_share_target!r}"
            )
        # below_target_exits for the inherited _snapshot_step:528 detect_h2h_
        # plateau. Under "plateau" the flipped at/above reading fires the D->E1
        # exit (reason "plateau_at_or_above_dominance_target"); under
        # "fixed_schedule" the inherited default is fine (exit_fires is ignored
        # by the run() override anyway).
        below_target_exits = (
            False if exit_mode == "plateau" else DEFAULT_BELOW_TARGET_EXITS
        )
        super().__init__(
            config,
            pool=pool,
            game_runner=game_runner,
            learner_policy=learner_policy,
            opponent_policies_factory=opponent_policies_factory,
            curriculum=curriculum,
            parity=parity,
            seed=seed,
            worker_factory=worker_factory,
            mana_draw_baseline=mana_draw_baseline,
            snapshot_cadence=snapshot_cadence,
            n_snap=n_snap,
            k_snap=k_snap,
            dominance_target=dominance_target,
            collapse_boost_factor=collapse_boost_factor,
            checkpoint_namer=checkpoint_namer,
            gauntlet_roster=gauntlet_roster,
            h2h_opponent_kind=h2h_opponent_kind,
            games_per_opponent_per_side=games_per_opponent_per_side,
            games_per_opponent_gauntlet=games_per_opponent_gauntlet,
            model=model,
            optimizer=optimizer,
            steps_per_update=steps_per_update,
            below_target_exits=below_target_exits,
            min_gain=min_gain,
        )
        # NEW instance attrs (the B8 __init__ already set self._aggregate_history
        # =[], self._h2h_history=[], self._last_rollout=None,
        # self._opponent_policies=None -- inherit those, do NOT reset).
        self.self_share_target = float(self_share_target)
        self.exit_mode = str(exit_mode)
        self.curriculum_off = bool(curriculum_off)
        self.e1_candidate_set = e1_candidate_set

    # ---- the per-update mix build (D1 + collapse monitor + D-D4 curriculum) --
    def _build_reweighted_mix(self) -> list[tuple[str, float]]:
        """Build the Block-D consolidation mix (D1, D-D1 ~0.50) with the collapse
        boost (the inherited D-B5 mana_draw-collapse monitor) + the D-D4
        curriculum (a NO-OP ``cap=0.0`` when ``curriculum_off``; the B4
        ``cap=0.25`` per-lane-loss reweight otherwise). Returns a renormalized
        ``[(name, weight)]`` summing to 1.0 with ``v5_snapshot`` MERGED into
        ``self`` (so the mix is dispatchable by A4 -- ``v5_snapshot`` is absent
        from ``POLICY_OPPONENT_KINDS`` / ``BLOCK_B_POLICY_OPPONENT_KINDS`` /
        ``RULE_AGENT_CODES``).

        Distinct from B8 ``_build_reweighted_mix`` (``block_b_league_driver.py:
        400-424``): B8 calls B3 ``build_block_b_opponent_mix`` (the Block-B
        0.05-cap weak-learner profile) + ``collapse_reweight_boost(factor)``
        splatted; D2 calls D1 ``build_block_d_opponent_mix`` (the D-D1
        ``self_share_target`` consolidation profile) with ``collapse_boost``
        passed DIRECTLY (D1 takes the float, not the B3 splat dict) + the
        D-D4 ``cap=0.0`` no-op when ``curriculum_off``.
        """
        md_rate = self._learner_mana_draw_rate()
        boost = self._collapse_boost_for(md_rate)
        mix = build_block_d_opponent_mix(
            self.pool,
            self_share_target=self.self_share_target,
            collapse_boost=boost,
        )
        # D-D4 curriculum: cap=0.0 -> NO-OP (every boost factor 1.0, renormalize
        # leaves a 1.0-sum mix unchanged); cap=0.25 -> the B4 per-lane-loss
        # reweight toward lanes the learner is losing to.
        if self.curriculum_off:
            reweighted = self.curriculum.reweight(mix, cap=0.0)
        else:
            reweighted = self.curriculum.reweight(mix, cap=0.25)
        # Merge the self-snapshot split into the dispatchable ``self`` identity
        # (IMPORTED helper; the merge preserves the total self+v5_snapshot
        # weight, so the mix still sums to 1.0).
        return _merge_self_snapshot_split(reweighted)

    # ---- the E1 candidate derivation (filled with post-D at exit) -----------
    def _derive_candidate_paths(self, best_path: str | None) -> list[str]:
        """Derive ``candidate_paths`` from the ``E1CandidateSet`` filled with the
        post-D ``best_path``. When the set is present it is FILLED (a NEW frozen
        ``E1CandidateSet`` via ``with_post_d``) and reassigned to
        ``self.e1_candidate_set`` so the driver attribute reflects the filled
        set after ``run``; the paths are ``[post_d, post_c3_best, post_b]`` with
        Nones dropped (order: post-D, post-C3, post-B per ``design.md:134``).
        When the set is None -> ``[best_path]`` (or ``[]`` if no best-ever)."""
        if self.e1_candidate_set is not None:
            filled = self.e1_candidate_set.with_post_d(best_path)
            self.e1_candidate_set = filled
            return [
                p
                for p in (
                    filled.post_d_path,
                    filled.post_c3_best_path,
                    filled.post_b_path,
                )
                if p is not None
            ]
        return [best_path] if best_path is not None else []

    # ---- the main loop (COPIED from B8 :564-628 + D2 exit branching) --------
    def run(self, n_updates: int) -> BlockDLeagueManifest:
        """Run ``n_updates`` Block-D consolidation updates + snapshot/gate/
        plateau every ``snapshot_cadence`` updates. Emits the D->E1 handoff at
        schedule end (D-D3 fixed-schedule, default) or on a dominant B7 plateau
        (D-D3 plateau).

        The per-update loop body is COPIED from B8 ``run()``
        (``block_b_league_driver.py:564-628``) -- B8 has no factored
        ``_run_update`` helper and ``block_b_league_driver.py`` is not editable,
        so the loop is duplicated + kept in sync (divergence risk noted,
        regression guard ``test_d2_loop_matches_b8_per_update_steps``). D2
        DIFFERENCES: (a) the mix is built via the overridden
        ``_build_reweighted_mix`` (D1 + merge); (b) a ``BlockDLeagueManifest``
        (NOT ``BlockBLeagueManifest``); (c) under fixed-schedule the B7
        ``exit_fires`` is IGNORED (no early return; the plateau is still
        MONITORED inside the inherited ``_snapshot_step``); (d) at exit the
        D->E1 handoff (``exited_to_e1`` + ``exit_verdict`` + ``candidate_paths``
        + ``aggregate_history``) is set.

        Uses the LIVE path (A4 ``run_live_self_play_update``), NOT
        ``train_rust_ppo_trace_files`` replay. The PPO optimizer step is
        MLX-gated inside A4 (``model`` / ``optimizer`` None -> stops after
        ``prepare_rust_ppo_batch``); the league loop itself runs without MLX.
        """
        if int(n_updates) < 0:
            raise ValueError("n_updates must be non-negative")
        opponent_policies = self._ensure_opponent_policies()
        # FRESH BlockDLeagueManifest (NOT BlockBLeagueManifest). aggregate_history
        # is already [] (fresh-seeded; NOT carried from CLoopManifest).
        manifest = BlockDLeagueManifest()

        for u in range(int(n_updates)):
            update_number = u + 1
            # (a)+(b)+(c) build the reweighted Block-D mix (D1 + merge).
            mix = self._build_reweighted_mix()

            # (d) B5 parity feedback: measured p1/p2 rates feed sample_learner_sides.
            p1_rate = float(self.parity.p1_score_rate())
            p2_rate = float(self.parity.p2_score_rate())

            # (e) A4 live update (Option 1: opponent_mix_parsed bypasses
            # parse_v5_opponent_mix so v4-orig-* pass through).
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
            slim_metrics["update_number"] = update_number
            manifest.update_metrics.append(slim_metrics)
            manifest.n_updates_run = update_number

            # (f) B4 curriculum update from the rollout's per-lane outcomes
            # (harmless when curriculum_off -- it accumulates per-lane loss that
            # the cap=0.0 reweight ignores).
            if rollout is not None:
                outcomes = extract_lane_outcomes(rollout)
                self.curriculum.update(outcomes)

            # (g) snapshot cadence -> B1 pool-add + B6 gate + B7 plateau
            # (MONITORED; the inherited _snapshot_step appends to
            # self._h2h_history + self._aggregate_history).
            if update_number % self.snapshot_cadence == 0:
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
                # EXIT BRANCHING (the load-bearing D2 divergence from B8):
                # under "plateau" a dominant B7 plateau fires the early D->E1
                # exit (reason "plateau_at_or_above_dominance_target" via
                # below_target_exits=False); under "fixed_schedule" the
                # exit_fires is IGNORED -- the loop continues to schedule end.
                if self.exit_mode == "plateau" and snap_record["exit_fires"]:
                    best_path = (
                        self.pool.best_ever.path
                        if self.pool.best_ever is not None
                        else None
                    )
                    manifest.exited_to_e1 = True
                    manifest.exit_verdict = snap_record["exit_verdict"]
                    manifest.best_ever_path = best_path
                    manifest.h2h_history = list(self._h2h_history)
                    manifest.aggregate_history = list(self._aggregate_history)
                    manifest.candidate_paths = self._derive_candidate_paths(
                        best_path
                    )
                    return manifest

        # AFTER the loop -- schedule-end for fixed_schedule, OR the fall-through
        # for plateau when no early exit fired. Set the D->E1 handoff.
        best_path = (
            self.pool.best_ever.path if self.pool.best_ever is not None else None
        )
        if self.exit_mode == "fixed_schedule":
            exit_verdict = {
                "reason": "block_d_schedule_complete",
                "n_updates_run": manifest.n_updates_run,
                "best_ever_path": best_path,
            }
        else:
            # plateau fall-through: the schedule completed without a dominant
            # plateau firing (e.g. the H2H series never reached K_snap+1 points
            # or never plateaued at/above the target).
            exit_verdict = {
                "reason": "block_d_schedule_complete_no_plateau",
                "n_updates_run": manifest.n_updates_run,
                "best_ever_path": best_path,
            }
        manifest.exited_to_e1 = True
        manifest.exit_verdict = exit_verdict
        manifest.best_ever_path = best_path
        manifest.h2h_history = list(self._h2h_history)
        manifest.aggregate_history = list(self._aggregate_history)
        manifest.candidate_paths = self._derive_candidate_paths(best_path)
        return manifest