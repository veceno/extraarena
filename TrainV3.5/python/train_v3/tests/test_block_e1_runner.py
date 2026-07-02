"""Synthetic tests for the Block E1 thin runner -- ``block_e1_runner.py`` (E-E12).

ALL collaborators are fakes: a fake ``GameRunner`` returning canned
``GameResult``s, a fake ``candidate_loader`` returning canned metadata, fake
``c2_client`` / ``scorecard_client``, a fake ``onnx_export_fn`` writing dummy
.onnx + .onnx.json files. NO real MLX/Rust/ONNX/rlhf_env DB/socket is touched --
the runner is the composition shell, + the threshold-table verdict logic is
unit-testable via the injected fakes. The prod wiring
(``infrastructure/config.py`` ``extra-lr-v5-max`` profile + the 4 retargeted
tiers) IS committed in the worktree, so ``ship_v5_winner``'s prod-wiring
verification passes without monkeypatching (the ONNX export is faked).

The fake ``GameRunner`` + ``_meta`` builder mirror the E3 test patterns
(``test_e1_tournament.py:82-127`` / ``:163-189``) so the per-lane score rates
are DETERMINISTIC + the threshold-table verdict is asserted EXACTLY.

Run: ``PYTHONPATH="$PWD:$PWD/TrainV3.5/python" python3 -m pytest
TrainV3.5/python/train_v3/tests/test_block_e1_runner.py -q``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# sys.path bootstrap: insert the worktree root (so ``rlhf_env`` /
# ``ai.train_v2`` / ``infrastructure`` resolve) AND the TrainV3.5/python parent
# (so ``train_v3.*`` resolves) when run via ``python -m pytest`` from the
# worktree root. Mirrors ``test_e1_tournament.py:34-41``.
_HERE = Path(__file__).resolve()
_TV3_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
from train_v3.block_d_league_driver import BlockDLeagueManifest  # noqa: E402
from train_v3.c_to_d_handoff import E1CandidateSet  # noqa: E402
from train_v3.e1_ship import ShipResult  # noqa: E402

import train_v3.block_e1_runner as runner  # noqa: E402
from train_v3.block_e1_runner import (  # noqa: E402
    build_e1_candidate_set_from_manifest,
    load_manifest,
    main,
    run_e1_pipeline,
    write_candidate_json,
)


# =============================================================================
# Fakes (mirror test_e1_tournament.py:82-189)
# =============================================================================
def _outcomes(rate: float, n: int = 100) -> list[str]:
    """Length-``n`` outcome list (wins + losses, NO draws) with score rate ==
    ``rate``. ``rate*n`` must be integer-rounded."""
    w = round(float(rate) * n)
    assert abs(w / n - float(rate)) < 1e-9, f"rate {rate} not expressible at n={n}"
    return ["win"] * w + ["loss"] * (n - w)


class _FakeGameRunner:
    """Deterministic fake ``GameRunner``: returns the next outcome from
    ``per_opponent_outcomes[opponent_kind]`` (cycling modulo). ``mana_draw_count``
    / ``eligible_turns`` are constant per game so the aggregate mana_draw rate is
    exactly ``mana_draw_count / eligible_turns``."""

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

    def play(self, opponent_kind: str, *, seed: int) -> GameResult:
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
    """Fake ``candidate_loader``: returns canned metadata. Mirrors the
    ``model_mlx.load_checkpoint`` return shape ``{"metadata": {...}}``."""

    def __init__(self, meta: dict) -> None:
        self.meta = dict(meta)
        self.calls: list[str] = []

    def __call__(self, path: str) -> dict:
        self.calls.append(path)
        return {"metadata": dict(self.meta)}


class _RaisingClient:
    """A fake c2_client / scorecard_client that RAISES on any method call -- used
    to assert the panel SOFT gate does NOT block ship."""

    def __getattr__(self, name: str):
        def _raise(*a, **k):
            raise RuntimeError(f"_RaisingClient.{name} raised (synthetic)")

        return _raise


def _per_opp(
    *,
    v4max: float = 0.75,
    random: float = 0.95,
    end_turn: float = 0.95,
    best_self: float = 0.55,
    n: int = 20,
) -> dict[str, list[str]]:
    """Build the per-opponent outcome map for the 4 anchor lanes; the 7 exploit
    kinds default to all-draws (their outcomes do NOT affect the verdict --
    exploit_resistance is metadata-sourced; they only feed the mana_draw
    aggregate which is constant per game).

    ``n=20`` matches the ``E1TournamentConfig.games_per_opponent=20`` default so
    the FakeGameRunner consumes the ENTIRE list (wins first) + the score rate is
    EXACT (no prefix-truncation skew)."""
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
    """The run-artifact metadata fields E3 reads from ``loaded["metadata"]``
    (``e1_tournament.py:528-566``)."""
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


def _fake_onnx_export_fn(checkpoint_path: str, output_path: str) -> str:
    """Fake ``onnx_export_fn``: writes a dummy .onnx + .onnx.json sidecar (NO real
    torch/onnx). The sidecar carries the V5 fingerprint so ``v5_detector`` would
    route it (not asserted here -- the ship test only asserts the bundle builds)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"dummy onnx (synthetic test)")
    sidecar = {
        "model_version": "v5_split_encoder_onnx_v1",
        "source_checkpoint": str(checkpoint_path),
        "obs_dim": 7128,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "placement_mode": "append_only",
        "inputs": ["observation", "action_features"],
        "outputs": ["logits", "value", "mana_draw_logit"],
        "mana_draw_head": True,
        "format": "v5",
    }
    sidecar_path = str(out) + ".json"
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh)
    return str(out)


# =============================================================================
# build_e1_candidate_set_from_manifest
# =============================================================================
def test_build_e1_candidate_set_from_manifest_full():
    """A dict with candidate_paths=[post_d.npz, post_c3.npz, post_b.npz] ->
    E1CandidateSet(post_d_path=post_d.npz, post_c3_best_path=post_c3.npz,
    post_b_path=post_b.npz)."""
    manifest = {
        "candidate_paths": ["post_d.npz", "post_c3.npz", "post_b.npz"],
        "best_ever_path": "best.npz",
        "exited_to_e1": True,
    }
    cs = build_e1_candidate_set_from_manifest(manifest)
    assert cs.post_d_path == "post_d.npz"
    assert cs.post_c3_best_path == "post_c3.npz"
    assert cs.post_b_path == "post_b.npz"


def test_build_e1_candidate_set_from_manifest_handles_none_and_short():
    """candidate_paths=[post_d.npz, None, post_b.npz] -> post_d_path=post_d.npz,
    post_c3_best_path=None, post_b_path=post_b.npz (None preserved in slot).
    candidate_paths=[only_post_d.npz] -> post_d_path=only_post_d.npz, others
    None (short list -> remaining fields None)."""
    cs = build_e1_candidate_set_from_manifest(
        {"candidate_paths": ["post_d.npz", None, "post_b.npz"]}
    )
    assert cs.post_d_path == "post_d.npz"
    assert cs.post_c3_best_path is None
    assert cs.post_b_path == "post_b.npz"

    cs2 = build_e1_candidate_set_from_manifest({"candidate_paths": ["only_post_d.npz"]})
    assert cs2.post_d_path == "only_post_d.npz"
    assert cs2.post_c3_best_path is None
    assert cs2.post_b_path is None


def test_build_e1_candidate_set_from_manifest_empty_falls_back_to_best_ever():
    """candidate_paths=[] + best_ever_path=best.npz -> post_d_path=best.npz
    (the fallback so a D-league that only produced best_ever still yields one
    candidate)."""
    cs = build_e1_candidate_set_from_manifest(
        {"candidate_paths": [], "best_ever_path": "best.npz"}
    )
    assert cs.post_d_path == "best.npz"
    assert cs.post_c3_best_path is None
    assert cs.post_b_path is None


def test_build_e1_candidate_set_from_manifest_empty_no_best_ever():
    """candidate_paths=[] + no best_ever -> all None (no candidates)."""
    cs = build_e1_candidate_set_from_manifest({"candidate_paths": [], "best_ever_path": None})
    assert cs.post_d_path is None
    assert cs.post_c3_best_path is None
    assert cs.post_b_path is None


def test_build_e1_candidate_set_from_manifest_accepts_object_and_dict():
    """Both a BlockDLeagueManifest object (with .candidate_paths) AND a dict (with
    [candidate_paths]) yield the same E1CandidateSet."""
    paths = ["post_d.npz", "post_c3.npz", "post_b.npz"]
    obj = BlockDLeagueManifest(
        candidate_paths=list(paths), best_ever_path="best.npz", exited_to_e1=True
    )
    d = {"candidate_paths": list(paths), "best_ever_path": "best.npz", "exited_to_e1": True}

    cs_obj = build_e1_candidate_set_from_manifest(obj)
    cs_dict = build_e1_candidate_set_from_manifest(d)
    assert cs_obj == cs_dict
    assert cs_obj.post_d_path == "post_d.npz"
    assert cs_obj.post_c3_best_path == "post_c3.npz"
    assert cs_obj.post_b_path == "post_b.npz"


# =============================================================================
# write_candidate_json
# =============================================================================
def test_write_candidate_json_writes_required_file(tmp_path):
    """write_candidate_json(tmp_dir, winner_path=win.npz) -> a candidate.json
    exists in tmp_dir with path=win.npz + marker=extra-lr-v5-max."""
    cpath = write_candidate_json(str(tmp_path), winner_path="win.npz")
    assert cpath == str(tmp_path / "candidate.json")
    assert (tmp_path / "candidate.json").exists()
    data = json.loads((tmp_path / "candidate.json").read_text(encoding="utf-8"))
    assert data["path"] == "win.npz"
    assert data["source_checkpoint"] == "win.npz"
    assert data["marker"] == "extra-lr-v5-max"
    assert data["created_by"] == "block_e1_runner"


# =============================================================================
# run_e1_pipeline -- the INJECTABLE composition (the core)
# =============================================================================
def _passing_manifest() -> dict:
    """A manifest with one passing candidate (post-D)."""
    return {
        "candidate_paths": ["post_d.npz", "post_c3.npz", "post_b.npz"],
        "best_ever_path": "post_d.npz",
        "exited_to_e1": True,
    }


def _passing_deps(tmp_path):
    """Build the passing-pipeline deps: fake game_runner + candidate_loader +
    real baseline + a ReleaseBundleConfig."""
    game_runner = _FakeGameRunner(_per_opp())
    candidate_loader = _FakeCandidateLoader(_meta())
    baseline = _baseline()
    bundle_config = runner.ReleaseBundleConfig(
        candidate_dir=str(tmp_path),
        output_dir=str(tmp_path / "bundle"),
    )
    return game_runner, candidate_loader, baseline, bundle_config


def test_run_e1_pipeline_ships_a_passing_winner(tmp_path):
    """Inject fakes (canned GameResults so a candidate passes the threshold table,
    canned metadata with throughput>12000, entropy>0.70, kl<0.12, no_bonus
    p1/p2/second>=0.70, mana_draw in band) + a fake onnx_export_fn (writes dummy
    .onnx + .onnx.json). Assert run_e1_pipeline returns a ShipResult (not None)
    + ship was reached + the candidate.json was written."""
    game_runner, candidate_loader, baseline, bundle_config = _passing_deps(tmp_path)
    c2_client = _RaisingClient()  # panel is SOFT; a raising client does not block
    scorecard_client = _RaisingClient()

    ship_result = run_e1_pipeline(
        _passing_manifest(),
        game_runner=game_runner,
        candidate_loader=candidate_loader,
        c2_client=c2_client,
        scorecard_client=scorecard_client,
        mana_draw_baseline=baseline,
        bundle_config=bundle_config,
        min_reviewers=3,
        min_battles=10,
        onnx_export_fn=_fake_onnx_export_fn,
    )

    assert ship_result is not None
    assert isinstance(ship_result, ShipResult)
    assert ship_result.winner_path == "post_d.npz"
    # candidate.json was written BEFORE ship (build_release_bundle requires it).
    assert (tmp_path / "candidate.json").exists()
    # the fake onnx_export_fn wrote the dummy onnx + sidecar into candidate_dir.
    assert (tmp_path / "extra-lr-v5-max.onnx").exists()
    assert (tmp_path / "extra-lr-v5-max.onnx.json").exists()


def test_run_e1_pipeline_no_ship_when_no_passer(tmp_path):
    """Fake game_runner returns outcomes where NO candidate passes (v4max H2H <
    0.70 for all) -> select_e1_winner returns None -> run_e1_pipeline returns
    None + ship NOT called."""
    game_runner = _FakeGameRunner(_per_opp(v4max=0.50))  # below 0.70 -> fail
    candidate_loader = _FakeCandidateLoader(_meta())
    baseline = _baseline()
    bundle_config = runner.ReleaseBundleConfig(
        candidate_dir=str(tmp_path), output_dir=str(tmp_path / "bundle")
    )

    ship_calls: list = []
    orig_ship = runner.ship_v5_winner

    def _spy_ship(*a, **k):
        ship_calls.append((a, k))
        return orig_ship(*a, **k)

    runner.ship_v5_winner = _spy_ship
    try:
        result = run_e1_pipeline(
            _passing_manifest(),
            game_runner=game_runner,
            candidate_loader=candidate_loader,
            c2_client=_RaisingClient(),
            scorecard_client=_RaisingClient(),
            mana_draw_baseline=baseline,
            bundle_config=bundle_config,
            onnx_export_fn=_fake_onnx_export_fn,
        )
    finally:
        runner.ship_v5_winner = orig_ship

    assert result is None
    assert ship_calls == []  # ship NOT called (no passer)


def test_run_e1_pipeline_no_ship_when_no_candidates(tmp_path):
    """Manifest with empty candidate_paths + no best_ever -> run_e1_pipeline
    returns None early (no tournament run)."""
    game_runner = _FakeGameRunner(_per_opp())
    candidate_loader = _FakeCandidateLoader(_meta())
    baseline = _baseline()
    bundle_config = runner.ReleaseBundleConfig(
        candidate_dir=str(tmp_path), output_dir=str(tmp_path / "bundle")
    )

    manifest = {"candidate_paths": [], "best_ever_path": None, "exited_to_e1": True}
    result = run_e1_pipeline(
        manifest,
        game_runner=game_runner,
        candidate_loader=candidate_loader,
        c2_client=_RaisingClient(),
        scorecard_client=_RaisingClient(),
        mana_draw_baseline=baseline,
        bundle_config=bundle_config,
        onnx_export_fn=_fake_onnx_export_fn,
    )
    assert result is None
    # the tournament was NOT run (candidate_loader never called).
    assert candidate_loader.calls == []


def test_run_e1_pipeline_panel_soft_does_not_block_ship(tmp_path):
    """A fake c2_client/scorecard_client that RAISES -> run_e1_pipeline still
    ships (the panel error is caught + logged, NOT propagated; the SOFT gate
    does not abort). Assert ship_v5_winner still called."""
    game_runner, candidate_loader, baseline, bundle_config = _passing_deps(tmp_path)
    c2_client = _RaisingClient()
    scorecard_client = _RaisingClient()

    ship_calls: list = []
    orig_ship = runner.ship_v5_winner

    def _spy_ship(*a, **k):
        ship_calls.append((a, k))
        return orig_ship(*a, **k)

    runner.ship_v5_winner = _spy_ship
    try:
        result = run_e1_pipeline(
            _passing_manifest(),
            game_runner=game_runner,
            candidate_loader=candidate_loader,
            c2_client=c2_client,
            scorecard_client=scorecard_client,
            mana_draw_baseline=baseline,
            bundle_config=bundle_config,
            min_reviewers=3,
            min_battles=10,
            onnx_export_fn=_fake_onnx_export_fn,
        )
    finally:
        runner.ship_v5_winner = orig_ship

    assert result is not None  # ship proceeded despite the panel raising
    assert len(ship_calls) == 1  # ship_v5_winner was called exactly once


def test_run_e1_pipeline_skip_panel(tmp_path):
    """run_panel=False -> run_e1_human_qa_panel NOT called, ship still
    proceeds."""
    game_runner, candidate_loader, baseline, bundle_config = _passing_deps(tmp_path)

    panel_calls: list = []
    orig_panel = runner.run_e1_human_qa_panel

    def _spy_panel(*a, **k):
        panel_calls.append((a, k))
        return orig_panel(*a, **k)

    runner.run_e1_human_qa_panel = _spy_panel
    try:
        result = run_e1_pipeline(
            _passing_manifest(),
            game_runner=game_runner,
            candidate_loader=candidate_loader,
            c2_client=_RaisingClient(),
            scorecard_client=_RaisingClient(),
            mana_draw_baseline=baseline,
            bundle_config=bundle_config,
            run_panel=False,
            onnx_export_fn=_fake_onnx_export_fn,
        )
    finally:
        runner.run_e1_human_qa_panel = orig_panel

    assert result is not None
    assert panel_calls == []  # panel NOT called (run_panel=False)


def test_composition_order(tmp_path):
    """Assert run_e1_tournament is called BEFORE select_e1_winner BEFORE
    ship_v5_winner (the composition order is load -> tournament -> select ->
    panel -> ship)."""
    call_order: list[str] = []

    orig_tournament = runner.run_e1_tournament
    orig_select = runner.select_e1_winner
    orig_ship = runner.ship_v5_winner
    orig_panel = runner.run_e1_human_qa_panel

    def _spy_tournament(*a, **k):
        call_order.append("tournament")
        return orig_tournament(*a, **k)

    def _spy_select(*a, **k):
        call_order.append("select")
        return orig_select(*a, **k)

    def _spy_panel(*a, **k):
        call_order.append("panel")
        return orig_panel(*a, **k)

    def _spy_ship(*a, **k):
        call_order.append("ship")
        return orig_ship(*a, **k)

    runner.run_e1_tournament = _spy_tournament
    runner.select_e1_winner = _spy_select
    runner.run_e1_human_qa_panel = _spy_panel
    runner.ship_v5_winner = _spy_ship
    try:
        run_e1_pipeline(
            _passing_manifest(),
            game_runner=_FakeGameRunner(_per_opp()),
            candidate_loader=_FakeCandidateLoader(_meta()),
            c2_client=_RaisingClient(),
            scorecard_client=_RaisingClient(),
            mana_draw_baseline=_baseline(),
            bundle_config=runner.ReleaseBundleConfig(
                candidate_dir=str(tmp_path), output_dir=str(tmp_path / "bundle")
            ),
            onnx_export_fn=_fake_onnx_export_fn,
        )
    finally:
        runner.run_e1_tournament = orig_tournament
        runner.select_e1_winner = orig_select
        runner.run_e1_human_qa_panel = orig_panel
        runner.ship_v5_winner = orig_ship

    # tournament -> select -> panel -> ship (the load step is before tournament;
    # it does not appear in call_order since build_e1_candidate_set_from_manifest
    # is not spied).
    assert call_order == ["tournament", "select", "panel", "ship"]


# =============================================================================
# main -- the thin CLI (argparse)
# =============================================================================
def test_main_parses_args_and_calls_pipeline(tmp_path, monkeypatch):
    """Invoke main with a fake argv + monkeypatch the operational factories +
    run_e1_pipeline to fakes (so main does not hit NotImplementedError) ->
    assert main returns 0 + run_e1_pipeline was called with the parsed args.

    Keep this test light (main is thin)."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "candidate_paths": ["post_d.npz"],
                "best_ever_path": "post_d.npz",
                "exited_to_e1": True,
            }
        ),
        encoding="utf-8",
    )
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    output_dir = tmp_path / "output"

    pipeline_calls: list = []

    class _FakeShipResult:
        winner_path = "post_d.npz"

    def _fake_pipeline(manifest, **kwargs):
        pipeline_calls.append((manifest, kwargs))
        return _FakeShipResult()  # truthy -> main returns 0

    monkeypatch.setattr(runner, "build_production_game_runner", lambda: object())
    monkeypatch.setattr(runner, "build_production_candidate_loader", lambda: lambda p: {})
    monkeypatch.setattr(runner, "build_production_c2_client", lambda: object())
    monkeypatch.setattr(runner, "build_production_scorecard_client", lambda: object())
    monkeypatch.setattr(runner, "run_e1_pipeline", _fake_pipeline)

    argv = [
        "--manifest", str(manifest_path),
        "--candidate-dir", str(candidate_dir),
        "--output-dir", str(output_dir),
        "--mana-draw-count", "40",
        "--eligible-turns", "100",
        "--min-reviewers", "2",
        "--min-battles", "5",
    ]
    rc = main(argv)

    assert rc == 0
    assert len(pipeline_calls) == 1
    manifest, kwargs = pipeline_calls[0]
    assert manifest["candidate_paths"] == ["post_d.npz"]
    assert kwargs["min_reviewers"] == 2
    assert kwargs["min_battles"] == 5
    assert kwargs["run_panel"] is True  # --skip-panel NOT passed
    # mana_draw_baseline built from the parsed --mana-draw-count / --eligible-turns.
    assert kwargs["mana_draw_baseline"].mana_draw_count == 40
    assert kwargs["mana_draw_baseline"].eligible_turns == 100
    # bundle_config built from the parsed --candidate-dir / --output-dir.
    assert kwargs["bundle_config"].candidate_dir == str(candidate_dir)
    assert kwargs["bundle_config"].output_dir == str(output_dir)