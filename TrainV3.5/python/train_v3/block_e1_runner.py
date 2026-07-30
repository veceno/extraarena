"""Block E1 thin in-worktree CLI runner -- ``block_e1_runner.py`` (E-E12).

V5-Max pipeline position: Block D COMPLETE (D1-D3 done) -> Block E1 (this file
is the E-E12 thin in-worktree runner + USER-run operational entry). It COMPOSES
the E1-E5 components READ-ONLY -- it does NOT mutate any E1-E5 file. The
composition order is:

  E1CandidateSet (reconstructed from the Block-D manifest)
    -> e1_tournament (E3 -- the tournament + final-acceptance threshold table)
    -> e1_human_qa_panel (E4 -- the SOFT human-QA gate, NEVER blocks ship)
    -> export_v5_checkpoint_to_onnx (E1 -- the ONNX export, injected)
    -> e1_ship (E5 -- export + bundle + register + verify)

The operational pieces (game_runner, candidate_loader, c2_client,
scorecard_client, mana_draw_baseline, the v5_policy) are USER-provided at
OPERATIONAL time -- the runner does NOT construct the MLX policy or the Rust
runner. ``run_e1_pipeline`` accepts them as INJECTED kwargs so the synthetic
tests inject fakes (NO real MLX/Rust/ONNX/rlhf_env DB/socket is touched). The
in-worktree ``main`` is the CLI skeleton; the operational factories
(``build_production_game_runner`` / ``build_production_candidate_loader`` /
``build_production_c2_client`` / ``build_production_scorecard_client``) are
USER-provided stubs that raise ``NotImplementedError`` referencing the
operational wiring (A4 ``rust_live_self_play``, ``model_mlx.load_checkpoint``,
rlhf_env MCP, ``JsonScorecardClient``). The real RUN is USER-executed per
E-E12; the in-worktree runner is the composition shell.

This module is TrainV3.5-side: it imports ``rlhf_env`` + ``ai.train_v2`` + the
E1-E5 ``train_v3`` components. It is NOT imported by prod -- no prod import of
``block_e1_runner`` exists (CONSTRAINT 1).

CLI pattern mirrors ``scripts/run_v5_acceptance.py`` (shebang, ROOT parents +
sys.path, ``def main(argv=None) -> int``, ``sys.exit(main())``) BUT uses
``TrainV3.5/python`` (NOT the broken ``TrainV3`` path,
``run_v5_acceptance.py:16``). The runner lives at
``TrainV3.5/python/train_v3/block_e1_runner.py`` (NOT ``scripts/``) so it is a
``train_v3`` module importable by the test via the PYTHONPATH bootstrap.

Run (operational, USER-executed):
  ``PYTHONPATH="$PWD:$PWD/TrainV3.5/python" python3 -m train_v3.block_e1_runner
   --manifest <BlockDLeagueManifest.json> --candidate-dir <dir>
   --output-dir <dir> --mana-draw-count <int> --eligible-turns <int>``

Run (synthetic tests):
  ``PYTHONPATH="$PWD:$PWD/TrainV3.5/python" python3 -m pytest
   TrainV3.5/python/train_v3/tests/test_block_e1_runner.py -q``
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

# sys.path bootstrap: insert the worktree root (so ``rlhf_env`` /
# ``ai.train_v2`` / ``infrastructure`` resolve) AND the TrainV3.5/python parent
# (so ``train_v3.*`` resolves) when run via ``python -m`` or imported by tests.
# ``__file__`` = .../<worktree>/TrainV3.5/python/train_v3/block_e1_runner.py
#   parents[0] = train_v3, parents[1] = python, parents[2] = TrainV3.5,
#   parents[3] = <worktree root>.
_HERE = Path(__file__).resolve()
_WORKTREE_ROOT = str(_HERE.parents[3])
_TV3_PARENT = str(_HERE.parents[1])
for _p in (_WORKTREE_ROOT, _TV3_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# E1-E5 train_v3 components (READ-ONLY composition -- the runner COMPOSES them,
# it does NOT mutate them). Relative imports so the module is a proper train_v3
# sibling (importable as ``train_v3.block_e1_runner``).
from .a_gate import ManaDrawBaseline, record_mana_draw_baseline
from .c_to_d_handoff import E1CandidateSet
from .e1_human_qa_panel import run_e1_human_qa_panel
from .e1_ship import ShipResult, ship_v5_winner
from .e1_tournament import (
    E1TournamentConfig,
    run_e1_tournament,
    select_e1_winner,
)
from .export_onnx_v5 import export_v5_checkpoint_to_onnx

# ai.train_v2 release bundle (ReleaseBundleConfig -- the bundle config the
# operational layer injects; the runner does NOT build the bundle itself,
# ship_v5_winner does). Imported for the type + the CLI construction.
from ai.train_v2.release_bundle import ReleaseBundleConfig

# rlhf_env McpCollectionClient -- Protocol reference only (the c2_client is
# INJECTED; the runner never constructs a real MCP client).
from rlhf_env.components.c2_collection_driver import McpCollectionClient  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "build_e1_candidate_set_from_manifest",
    "write_candidate_json",
    "run_e1_pipeline",
    "load_manifest",
    "main",
]


# =============================================================================
# build_e1_candidate_set_from_manifest -- reconstruct the E1CandidateSet from
# the flat BlockDLeagueManifest.candidate_paths list ([post-D, post-C3, post-B]
# ordered). NO helper in c_to_d_handoff builds an E1CandidateSet from a flat
# list -- the runner reconstructs it manually.
# =============================================================================
def build_e1_candidate_set_from_manifest(manifest: Any) -> E1CandidateSet:
    """Reconstruct the ``E1CandidateSet`` from a ``BlockDLeagueManifest``.

    Accepts EITHER a dict (loaded from JSON -- the CLI path) OR a
    ``BlockDLeagueManifest`` object (tests may pass either). Reads
    ``candidate_paths`` (ordered [post-D, post-C3, post-B]) + the fallback
    ``best_ever_path``.

    Mapping is POSITIONAL and ORDER-PRESERVING. Two shapes are accepted:
      * A hand-built dict MAY carry in-slot ``None`` entries (a ``None`` maps to
        a ``None`` field, "dropping" that candidate from the slot).
      * A REAL ``BlockDLeagueManifest`` already DROPS ``None`` entries via
        ``_derive_candidate_paths`` (block_d_league_driver.py), so its
        ``candidate_paths`` is a compact ordered list -- the positional mapping
        is order-preserving but NOT slot-faithful when post-D was ``None``
        (e.g. a real manifest with post-D=None + post-C3 + post-B yields
        ``candidate_paths=[post_c3, post_b]`` which maps post_c3->post_d_path).
        This is functionally harmless: downstream consumers
        (``_candidate_paths`` in e1_tournament.py, ``_iter_candidate_paths`` in
        e1_human_qa_panel.py) iterate the ``E1CandidateSet`` fields in order with
        None-drop + dedup, so the ORDER of candidates fed to the tournament/panel
        is preserved exactly -- no consumer reads slot labels semantically
        (``E1CandidateReport.candidate_path`` + ``ShipResult.winner_path`` ARE
        the path itself, not the slot).
      * ``candidate_paths[0]`` -> ``post_d_path``
      * ``candidate_paths[1]`` -> ``post_c3_best_path``
      * ``candidate_paths[2]`` -> ``post_b_path``
      * short lists (fewer than 3) -> remaining fields None.
      * empty ``candidate_paths`` -> fall back to ``best_ever_path`` as
        ``post_d_path`` (so a D-league that only produced ``best_ever`` still
        yields one candidate).

    Returns an ``E1CandidateSet`` (all fields may be None when there are no
    candidates -- the caller checks + returns a NO-SHIP early).
    """
    # Accept dict (JSON-loaded) OR object (BlockDLeagueManifest).
    if isinstance(manifest, dict):
        candidate_paths = list(manifest.get("candidate_paths", []) or [])
        best_ever_path = manifest.get("best_ever_path")
    else:
        candidate_paths = list(getattr(manifest, "candidate_paths", []) or [])
        best_ever_path = getattr(manifest, "best_ever_path", None)

    # Empty candidate_paths -> fall back to best_ever_path as the single
    # post-D candidate (a D-league that only produced best_ever still yields
    # one candidate).
    if not candidate_paths:
        return E1CandidateSet(post_d_path=best_ever_path)

    # Positional mapping with None preserved (a None entry -> None field).
    post_d_path = candidate_paths[0] if len(candidate_paths) > 0 else None
    post_c3_best_path = candidate_paths[1] if len(candidate_paths) > 1 else None
    post_b_path = candidate_paths[2] if len(candidate_paths) > 2 else None

    return E1CandidateSet(
        post_d_path=post_d_path,
        post_c3_best_path=post_c3_best_path,
        post_b_path=post_b_path,
    )


# =============================================================================
# write_candidate_json -- write the candidate.json the release bundle REQUIRES
# (build_release_bundle raises FileNotFoundError without it). ship_v5_winner
# does NOT write it -- this is the runner's job.
# =============================================================================
def write_candidate_json(
    candidate_dir: str,
    *,
    winner_path: str,
    source_checkpoint: Optional[str] = None,
) -> str:
    """Write a ``candidate.json`` into ``candidate_dir`` (the release bundle
    requires it -- ``build_release_bundle`` raises ``FileNotFoundError`` if it
    is missing).

    The JSON carries:
      * ``path`` -- the winning checkpoint path.
      * ``source_checkpoint`` -- the source checkpoint (defaults to
        ``winner_path``).
      * ``marker`` -- ``"extra-lr-v5-max"`` (the V5-Max release marker).
      * ``created_by`` -- ``"block_e1_runner"`` (the runner provenance).

    Returns the ``candidate.json`` path (``<candidate_dir>/candidate.json``).
    """
    os.makedirs(candidate_dir, exist_ok=True)
    candidate_json_path = os.path.join(candidate_dir, "candidate.json")
    payload = {
        "path": str(winner_path),
        "source_checkpoint": str(source_checkpoint or winner_path),
        "marker": "extra-lr-v5-max",
        "created_by": "block_e1_runner",
    }
    with open(candidate_json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return candidate_json_path


# =============================================================================
# run_e1_pipeline -- the INJECTABLE composition (the core; tests inject fakes
# here). The candidate_loader, game_runner, c2_client, scorecard_client are ALL
# injected (the runner does NOT construct the MLX policy or the Rust runner).
# mana_draw_baseline is injected (operational measurement). bundle_config is
# injected (ReleaseBundleConfig).
# =============================================================================
def run_e1_pipeline(
    manifest: Any,
    *,
    game_runner: Any,
    candidate_loader: Callable[[str], dict],
    c2_client: Any,
    scorecard_client: Any,
    mana_draw_baseline: ManaDrawBaseline,
    bundle_config: ReleaseBundleConfig,
    min_reviewers: int = 3,
    min_battles: int = 10,
    no_bonus_benchmark_json_path: Optional[str] = None,
    battles_per_series: int = 1000,
    run_panel: bool = True,
    onnx_export_fn: Callable[[str, str], str] = export_v5_checkpoint_to_onnx,
) -> Optional[ShipResult]:
    """Run the E1 tournament + (optional) human-QA panel + ship the winner.

    Composition order: load manifest -> build candidate_set -> run tournament
    (E3) -> select winner (E3) -> run human-QA panel (E4, SOFT -- never blocks)
    -> write candidate.json -> ship (E5).

    Steps:
      (a) build ``candidate_set`` from ``manifest``; if no candidates (all
          None) -> log + return None (NO-SHIP, no candidates).
      (b) build ``E1TournamentConfig`` (defaulted games_per_opponent /
          gauntlet_roster / floors).
      (c) ``reports = run_e1_tournament(config, game_runner=...,
          candidate_loader=...)``.
      (d) ``winner_report = select_e1_winner(reports)``; if None -> log NO-SHIP
          (no passer) + return None.
      (e) if ``run_panel``: call ``run_e1_human_qa_panel`` (SOFT -- log the
          verdicts, do NOT let it block; wrap in try/except so a panel error
          does not abort the ship -- log + continue).
      (f) ``write_candidate_json`` (the release bundle requires it; ship_v5_winner
          does NOT write it).
      (g) ``ship_v5_winner(winner_report, onnx_export_fn=onnx_export_fn,
          bundle_config=bundle_config)``.
      (h) return the ``ShipResult``.

    NO-SHIP paths (no candidates, no passer) return None -- they do NOT raise
    (a no-ship is a valid tournament outcome).
    """
    # (a) build the candidate set from the manifest.
    candidate_set = build_e1_candidate_set_from_manifest(manifest)
    if not _has_candidates(candidate_set):
        logger.info(
            "[e1-runner] NO-SHIP: no candidates (manifest produced no "
            "candidate_paths and no best_ever_path)"
        )
        return None

    # (b) build the tournament config (defaulted -- the runner uses the default
    # gauntlet_roster / games_per_opponent / floors; it does NOT build the roster).
    config = E1TournamentConfig(
        candidate_set=candidate_set,
        mana_draw_baseline=mana_draw_baseline,
        no_bonus_benchmark_json_path=no_bonus_benchmark_json_path,
    )

    # (c) run the tournament.
    reports = run_e1_tournament(
        config, game_runner=game_runner, candidate_loader=candidate_loader
    )

    # (d) select the winner (NO-SHIP when no passer).
    winner_report = select_e1_winner(reports)
    if winner_report is None:
        logger.info(
            "[e1-runner] NO-SHIP: no candidate passed the final-acceptance "
            "threshold table (select_e1_winner returned None)"
        )
        return None

    # (e) human-QA panel (SOFT -- never blocks ship). Wrap in try/except so a
    # panel error does NOT abort the ship (log + continue; the SOFT gate is
    # advisory only per E-E8).
    if run_panel:
        try:
            verdicts = run_e1_human_qa_panel(
                candidate_set,
                c2_client=c2_client,
                scorecard_client=scorecard_client,
                min_reviewers=int(min_reviewers),
                min_battles=int(min_battles),
                battles_per_series=int(battles_per_series),
            )
            logger.info(
                "[e1-runner] human-QA panel verdicts (SOFT -- does not block): %s",
                {
                    path: getattr(v, "verdict", str(v))
                    for path, v in verdicts.items()
                },
            )
        except Exception as exc:  # noqa: BLE001 -- SOFT gate: never abort ship
            logger.warning(
                "[e1-runner] human-QA panel raised (SOFT gate -- caught, "
                "ship proceeds): %s", exc
            )

    # (f) write candidate.json (the release bundle REQUIRES it; ship_v5_winner
    # does NOT write it -- build_release_bundle raises FileNotFoundError without
    # it).
    write_candidate_json(
        bundle_config.candidate_dir,
        winner_path=winner_report.candidate_path,
        source_checkpoint=winner_report.candidate_path,
    )

    # (g) ship the winner (E5 -- export ONNX + build bundle + register detector
    # + verify prod wiring).
    ship_result = ship_v5_winner(
        winner_report,
        onnx_export_fn=onnx_export_fn,
        bundle_config=bundle_config,
    )

    # (h) return the ship result.
    return ship_result


def _has_candidates(candidate_set: E1CandidateSet) -> bool:
    """True iff at least one of the three candidate paths is non-None."""
    return any(
        p is not None
        for p in (
            candidate_set.post_d_path,
            candidate_set.post_c3_best_path,
            candidate_set.post_b_path,
        )
    )


# =============================================================================
# load_manifest -- load a BlockDLeagueManifest JSON from disk.
# =============================================================================
def load_manifest(path: str) -> dict:
    """Load a ``BlockDLeagueManifest`` JSON from disk (``json.load``).

    Returns a dict with ``candidate_paths`` / ``best_ever_path`` /
    ``exited_to_e1`` (the runner operates on the dict form;
    ``build_e1_candidate_set_from_manifest`` accepts the dict).
    """
    with open(str(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


# =============================================================================
# Operational factory stubs -- USER-provided (the real RUN is USER-executed per
# E-E12). The in-worktree stubs raise NotImplementedError referencing the
# operational wiring. The synthetic tests do NOT call main (they test
# run_e1_pipeline with fakes directly); test_main_parses_args_and_calls_pipeline
# monkeypatches these stubs + run_e1_pipeline to fakes.
# =============================================================================
def build_production_game_runner() -> Any:
    """Build the production A4 ``rust_live_self_play`` game runner (USER-provided).

    Operational wiring: the A4 ``rust_live_self_play`` entry point
    (``rust_live_self_play.py`` ``run_live_self_play_update`` /
    ``collect_rust_live_rollout``) wrapped as an ``a_gate.GameRunner``
    (``play(opponent_kind, *, seed) -> GameResult``) playing a real game on the
    Rust ``ArenaEnv`` + harvesting the outcome + mana_draw channels.

    The in-worktree stub raises ``NotImplementedError`` -- the real RUN is
    USER-executed per E-E12 (the runner is the composition shell, NOT the
    operational wiring).
    """
    raise NotImplementedError(
        "build_production_game_runner: the operational A4 rust_live_self_play "
        "game runner is USER-provided at operational time (E-E12). Wire it in "
        "the operational layer (rust_live_self_play.py "
        "run_live_self_play_update / collect_rust_live_rollout wrapped as an "
        "a_gate.GameRunner). The in-worktree runner is the composition shell."
    )


def build_production_candidate_loader() -> Callable[[str], dict]:
    """Build the production candidate loader (USER-provided).

    Operational wiring: ``e1_tournament.make_default_candidate_loader(v5_policy)``
    where ``v5_policy`` is a ``V5ActionConditionedPolicy`` (MLX). The loader
    calls ``ai.train_v2.model_mlx.load_checkpoint(path, policy)`` + returns
    ``{"metadata": {...}}`` (the run-artifact fields E3 reads: throughput /
    entropy / max_abs_kl / no_bonus p1/p2/second / h2h_vs_self_snapshot_history
    / p1_p2_gap / no_assist_score_rate / exploit_resistance_score_rate).

    The in-worktree stub raises ``NotImplementedError`` -- the real RUN is
    USER-executed per E-E12 (the runner does NOT construct the MLX policy).
    """
    raise NotImplementedError(
        "build_production_candidate_loader: the operational candidate loader "
        "is USER-provided at operational time (E-E12). Construct the V5 "
        "V5ActionConditionedPolicy (MLX) + call "
        "e1_tournament.make_default_candidate_loader(v5_policy) in the "
        "operational layer. The in-worktree runner does NOT construct the MLX "
        "policy."
    )


def build_production_c2_client() -> Any:
    """Build the production C2 MCP collection client (USER-provided).

    Operational wiring: the rlhf_env MCP ``McpCollectionClient`` (the C2
    observer surface, ``rlhf_env/components/c2_collection_driver.py`` Protocol:
    list_v5_groups / get_v5_dataset_summary / get_v5_trace /
    validate_v5_traces). Human matches are owned by the web process; this
    client only harvests completed groups from the shared sessions directory.

    The in-worktree stub raises ``NotImplementedError`` -- the real RUN is
    USER-executed per E-E12.
    """
    raise NotImplementedError(
        "build_production_c2_client: the operational rlhf_env MCP "
        "McpCollectionClient is USER-provided at operational time (E-E12). Wire "
        "the rlhf_env MCP client in the operational layer. The in-worktree "
        "runner is the composition shell."
    )


def build_production_scorecard_client() -> Any:
    """Build the production reviewer scorecard client (USER-provided).

    Operational wiring: ``e1_human_qa_panel.JsonScorecardClient(path)`` (the
    JSON-file-backed ``ReviewerScorecardClient``, ``e1_human_qa_panel.py:178``)
    -- the USER-run operational entry that reads/writes a scorecards JSON file.

    The in-worktree stub raises ``NotImplementedError`` -- the real RUN is
    USER-executed per E-E12.
    """
    raise NotImplementedError(
        "build_production_scorecard_client: the operational "
        "ReviewerScorecardClient is USER-provided at operational time (E-E12). "
        "Construct e1_human_qa_panel.JsonScorecardClient(path) in the "
        "operational layer. The in-worktree runner is the composition shell."
    )


# =============================================================================
# main -- the thin CLI (argparse). The operational factories are USER-provided
# stubs that raise NotImplementedError (the real RUN is USER-executed per E-E12;
# the in-worktree runner is the composition shell). The synthetic tests test
# run_e1_pipeline with fakes (NOT main).
# =============================================================================
def main(argv: Optional[list[str]] = None) -> int:
    """Thin CLI skeleton for the E-E12 USER-run operational entry.

    Parses args, loads the manifest, builds the ManaDrawBaseline +
    ReleaseBundleConfig, constructs the operational pieces via the factory
    stubs, + calls ``run_e1_pipeline``. The operational factories
    (``build_production_game_runner`` / ``build_production_candidate_loader`` /
    ``build_production_c2_client`` / ``build_production_scorecard_client``)
    raise ``NotImplementedError`` in the in-worktree shell -- the real RUN is
    USER-executed per E-E12 (the USER wires the operational factories in the
    operational layer).

    Returns 0 on ship (a ``ShipResult`` was produced), 1 on NO-SHIP (no
    candidates / no passer -- ``run_e1_pipeline`` returned None).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Block E1 thin runner -- the E-E12 USER-run operational entry. "
            "Composes E1CandidateSet -> e1_tournament -> e1_human_qa_panel "
            "-> export_v5_checkpoint_to_onnx -> e1_ship. The operational "
            "pieces (game_runner, candidate_loader, c2_client, "
            "scorecard_client, mana_draw_baseline) are USER-provided."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the BlockDLeagueManifest JSON (the D-league exit manifest).",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        required=True,
        help="Path to the candidate dir (where the ONNX + sidecar + candidate.json land).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the release bundle output dir.",
    )
    parser.add_argument(
        "--mana-draw-count",
        type=int,
        required=True,
        help="Production-measured mana_draw count (the Q4 human baseline B numerator).",
    )
    parser.add_argument(
        "--eligible-turns",
        type=int,
        required=True,
        help="Production-measured eligible turns (the Q4 human baseline B denominator).",
    )
    parser.add_argument(
        "--min-reviewers",
        type=int,
        default=3,
        help="Minimum reviewers for the human-QA panel coverage stop (default 3).",
    )
    parser.add_argument(
        "--min-battles",
        type=int,
        default=10,
        help="Minimum battles for the human-QA panel coverage stop (default 10).",
    )
    parser.add_argument(
        "--no-bonus-benchmark",
        type=Path,
        default=None,
        help="Path to the V4-max pre-baked no_bonus benchmark JSON (SOFT advisory).",
    )
    parser.add_argument(
        "--skip-panel",
        action="store_true",
        help="Skip the human-QA panel (run_panel=False).",
    )
    parser.add_argument(
        "--battles-per-series",
        type=int,
        default=1000,
        help="Battles per C2 series for the human-QA panel (default 1000).",
    )
    args = parser.parse_args(argv)

    # Load the manifest (dict form -- the runner operates on the dict).
    manifest = load_manifest(str(args.manifest))

    # Build the ManaDrawBaseline (production-measured via the CLI args).
    mana_draw_baseline = record_mana_draw_baseline(
        int(args.mana_draw_count),
        int(args.eligible_turns),
    )

    # Build the ReleaseBundleConfig.
    bundle_config = ReleaseBundleConfig(
        candidate_dir=str(args.candidate_dir),
        output_dir=str(args.output_dir),
    )

    # Operational factories -- USER-provided stubs (the real RUN is USER-executed
    # per E-E12). The in-worktree stubs raise NotImplementedError referencing the
    # operational wiring. The USER wires these in the operational layer.
    game_runner = build_production_game_runner()
    candidate_loader = build_production_candidate_loader()
    c2_client = build_production_c2_client()
    scorecard_client = build_production_scorecard_client()

    no_bonus_benchmark_json_path = (
        str(args.no_bonus_benchmark) if args.no_bonus_benchmark is not None else None
    )

    ship_result = run_e1_pipeline(
        manifest,
        game_runner=game_runner,
        candidate_loader=candidate_loader,
        c2_client=c2_client,
        scorecard_client=scorecard_client,
        mana_draw_baseline=mana_draw_baseline,
        bundle_config=bundle_config,
        min_reviewers=int(args.min_reviewers),
        min_battles=int(args.min_battles),
        no_bonus_benchmark_json_path=no_bonus_benchmark_json_path,
        battles_per_series=int(args.battles_per_series),
        run_panel=not bool(args.skip_panel),
    )

    return 0 if ship_result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
