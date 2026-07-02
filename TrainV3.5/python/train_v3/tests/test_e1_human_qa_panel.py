"""Synthetic tests for Block E1 component E4 -- ``e1_human_qa_panel.py``.

All tests are SYNTHETIC: a fake ``McpCollectionClient`` returning canned C2
battle results + a fake ``ReviewerScorecardClient`` returning canned
scorecards + a tmp JSON scorecards file. NO real rlhf_env server, NO real V5
checkpoint, NO real reviewers. The verdict derivation logic is unit-testable
WITHOUT MLX/Rust/ONNX.

Run:
  PYTHONPATH="/path/to/worktree:/path/to/worktree/TrainV3.5/python" python3 \\
      -m pytest TrainV3.5/python/train_v3/tests/test_e1_human_qa_panel.py -q
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

# sys.path bootstrap: insert the worktree root (so ``rlhf_env.*`` resolves) AND
# the TrainV3.5/python parent (so ``train_v3.*`` resolves) when run via
# ``python -m pytest`` from the worktree root. Mirrors the Block D / E3 test
# pattern (``test_e1_tournament.py:29-41``).
_HERE = Path(__file__).resolve()
_TV3_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# .../glm-TrainV3.5Prep/TrainV3.5/python/train_v3/tests/test_e1_human_qa_panel.py
# worktree root = 4 levels up.
_WORKTREE_ROOT = os.path.abspath(str(_HERE.parents[4]))
for _p in (_TV3_PARENT, _WORKTREE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rlhf_env.components.c2_collection_driver import (  # noqa: E402
    C2CollectionResult,
    McpCollectionClient,
)
from train_v3.c_to_d_handoff import E1CandidateSet  # noqa: E402
from train_v3.e1_human_qa_panel import (  # noqa: E402
    MANA_DRAW_BLIND_SCOPE,
    E1HumanQAPanelDriver,
    HumanQAVerdict,
    JsonScorecardClient,
    ReviewerScorecardClient,
    aggregate_scorecards,
    derive_verdict,
    run_e1_human_qa_panel,
)


# =============================================================================
# Fakes
# =============================================================================
class _FakeC2Client:
    """A minimal ``McpCollectionClient`` Protocol implementation for tests.

    ``collect()`` drives ``start_series`` / ``next_battle`` /
    ``get_v5_dataset_summary`` / ``get_v5_trace`` / ``list_battles``. The fake
    returns canned responses so ``C2CollectionResult.battle_count`` is
    controlled (coverage floor). It SPIES on the deployed candidate path via
    ``start_series`` -> ``p2_model.path``.
    """

    def __init__(self, *, battle_count_per_series: int = 10) -> None:
        self.battle_count_per_series = int(battle_count_per_series)
        self.deployed_paths: list[str] = []
        self.start_calls: int = 0

    # -- McpCollectionClient surface ----------------------------------------
    def start_series(self, spec):
        self.start_calls += 1
        path = spec.get("p2_model", {}).get("path") if isinstance(spec, dict) else None
        if path is not None:
            self.deployed_paths.append(path)
        return {"group_id": f"g-{self.start_calls}", "v5_trace_ok": True}

    def next_battle(self, group_id):
        # Signal series complete immediately; battles are counted via the
        # summary in _harvest_group.
        return {"status": "series_complete"}

    def list_v5_groups(self, *args, **kwargs):
        return {"groups": []}

    def get_v5_dataset_summary(self, group_id):
        return {
            "battles_finished": self.battle_count_per_series,
            "v5_trace_ok_count": self.battle_count_per_series,
            "battle_ids": [f"{group_id}-b{i}" for i in range(self.battle_count_per_series)],
        }

    def get_v5_trace(self, group_id, battle_id, what):
        # No mana_draw rows in the fake (the panel is mana_draw-BLIND; the
        # mana_draw axis is NOT exercised here).
        return {"data": []}

    def list_battles(self, group_id):
        return [{"battle_id": f"{group_id}-b{i}"} for i in range(self.battle_count_per_series)]


class _FakeScorecardClient:
    """A minimal ``ReviewerScorecardClient`` Protocol implementation for tests.

    Holds an in-memory list of scorecard rows; ``list_scorecards`` returns the
    canned rows for a candidate. ``submit_scorecard`` appends.
    """

    def __init__(self, rows_by_candidate: dict | None = None) -> None:
        self.rows: list[dict] = []
        if rows_by_candidate:
            for cand, rows in rows_by_candidate.items():
                for r in rows:
                    row = dict(r)
                    row.setdefault("candidate_path", cand)
                    self.rows.append(row)

    def submit_scorecard(self, candidate_path, *, reviewer_id, difficulty_score, harder_than_baseline, notes=""):
        row = {
            "candidate_path": candidate_path,
            "reviewer_id": reviewer_id,
            "difficulty_score": float(difficulty_score),
            "harder_than_baseline": bool(harder_than_baseline),
            "notes": str(notes),
        }
        self.rows.append(row)
        return row

    def list_scorecards(self, candidate_path):
        return [r for r in self.rows if r.get("candidate_path") == candidate_path]


def _scorecard(reviewer_id, difficulty_score, harder_than_baseline, notes=""):
    return {
        "reviewer_id": reviewer_id,
        "difficulty_score": float(difficulty_score),
        "harder_than_baseline": bool(harder_than_baseline),
        "notes": str(notes),
    }


# =============================================================================
# Tests
# =============================================================================
def test_human_qa_verdict_schema_roundtrip():
    """(1) Construct a HumanQAVerdict, assert all fields; the ``.verdict``
    attribute is present (E3 duck-type interop, ``tests/test_e1_tournament.py:
    130-136`` relies on ``.verdict``)."""
    v = HumanQAVerdict(
        candidate_path="/tmp/cand.ckpt",
        n_battles=20,
        n_reviewers=3,
        mean_difficulty_score=4.2,
        n_harder_than_baseline=2,
        verdict="harder",
        freeform_notes="feels tougher",
        stop_condition_met=True,
    )
    assert v.candidate_path == "/tmp/cand.ckpt"
    assert v.n_battles == 20
    assert v.n_reviewers == 3
    assert v.mean_difficulty_score == 4.2
    assert v.n_harder_than_baseline == 2
    assert v.verdict == "harder"  # E3 duck-type interop surface
    assert v.freeform_notes == "feels tougher"
    assert v.stop_condition_met is True
    # frozen dataclass
    with pytest.raises(Exception):
        v.verdict = "easier"  # type: ignore[misc]
    # default freeform_notes == ""
    v2 = HumanQAVerdict("p", 1, 1, 3.0, 0, "comparable")
    assert v2.freeform_notes == ""
    assert v2.stop_condition_met is False


def test_stop_condition_fires_on_coverage():
    """(2) stop_condition_met True when n_reviewers >= min AND n_battles >= min;
    False when EITHER is below the min (-> 'inconclusive')."""
    verdict, stop = derive_verdict(
        mean_difficulty_score=4.5,
        n_harder_than_baseline=3,
        n_reviewers=3,
        n_battles=10,
        min_reviewers=3,
        min_battles=10,
    )
    assert stop is True
    assert verdict == "harder"
    # reviewers below min -> inconclusive + stop False
    v2, stop2 = derive_verdict(
        mean_difficulty_score=4.5,
        n_harder_than_baseline=3,
        n_reviewers=2,
        n_battles=10,
        min_reviewers=3,
        min_battles=10,
    )
    assert stop2 is False
    assert v2 == "inconclusive"
    # battles below min -> inconclusive + stop False
    v3, stop3 = derive_verdict(
        mean_difficulty_score=4.5,
        n_harder_than_baseline=3,
        n_reviewers=3,
        n_battles=5,
        min_reviewers=3,
        min_battles=10,
    )
    assert stop3 is False
    assert v3 == "inconclusive"


def test_harder_verdict_is_soft_pass():
    """(3) mean >= 4.0 OR n_harder >= ceil(n_reviewers/2) -> 'harder' -> SOFT
    pass (E4 does NOT raise/block on a harder verdict)."""
    # mean >= 4.0 path
    v1, stop1 = derive_verdict(
        mean_difficulty_score=4.1,
        n_harder_than_baseline=0,
        n_reviewers=3,
        n_battles=10,
        min_reviewers=3,
        min_battles=10,
    )
    assert v1 == "harder"
    assert stop1 is True
    # n_harder >= ceil(n_reviewers/2) path (mean < 4.0)
    v2, stop2 = derive_verdict(
        mean_difficulty_score=3.5,
        n_harder_than_baseline=2,  # ceil(3/2) == 2
        n_reviewers=3,
        n_battles=10,
        min_reviewers=3,
        min_battles=10,
    )
    assert v2 == "harder"
    assert stop2 is True
    # The panel does NOT raise/block on 'harder' -- run_e1_human_qa_panel
    # returns the verdict map WITHOUT raising.
    cands = E1CandidateSet(post_d_path="/tmp/h.ckpt")
    c2 = _FakeC2Client(battle_count_per_series=10)
    sc = _FakeScorecardClient({
        "/tmp/h.ckpt": [
            _scorecard("r1", 4.5, True),
            _scorecard("r2", 4.2, True),
            _scorecard("r3", 4.1, True),
        ]
    })
    out = run_e1_human_qa_panel(
        cands, c2_client=c2, scorecard_client=sc, min_reviewers=3, min_battles=10
    )
    assert out["/tmp/h.ckpt"].verdict == "harder"


def test_easier_verdict_is_soft_warn_not_hard_fail():
    """(4) mean < 3.0 -> 'easier' -> SOFT warn (the panel does NOT raise, does
    NOT block the ship; run_e1_human_qa_panel returns the verdict map WITHOUT
    raising)."""
    cands = E1CandidateSet(post_d_path="/tmp/e.ckpt")
    c2 = _FakeC2Client(battle_count_per_series=10)
    sc = _FakeScorecardClient({
        "/tmp/e.ckpt": [
            _scorecard("r1", 2.0, False),
            _scorecard("r2", 2.5, False),
            _scorecard("r3", 1.5, False),
        ]
    })
    out = run_e1_human_qa_panel(
        cands, c2_client=c2, scorecard_client=sc, min_reviewers=3, min_battles=10
    )
    v = out["/tmp/e.ckpt"]
    assert v.verdict == "easier"
    assert v.stop_condition_met is True  # coverage met
    # SOFT warn -- no exception raised (the call returned).


def test_inconclusive_when_coverage_not_met():
    """(5) n_reviewers < min OR n_battles < min -> 'inconclusive' +
    stop_condition_met False."""
    # reviewers below min
    cands = E1CandidateSet(post_d_path="/tmp/i1.ckpt")
    c2 = _FakeC2Client(battle_count_per_series=10)
    sc = _FakeScorecardClient({
        "/tmp/i1.ckpt": [_scorecard("r1", 4.5, True)]  # only 1 reviewer
    })
    out = run_e1_human_qa_panel(
        cands, c2_client=c2, scorecard_client=sc, min_reviewers=3, min_battles=10
    )
    v1 = out["/tmp/i1.ckpt"]
    assert v1.verdict == "inconclusive"
    assert v1.stop_condition_met is False
    # battles below min -- set min_battles above the fake's per-series count.
    cands2 = E1CandidateSet(post_d_path="/tmp/i2.ckpt")
    c2b = _FakeC2Client(battle_count_per_series=5)
    scb = _FakeScorecardClient({
        "/tmp/i2.ckpt": [
            _scorecard("r1", 4.5, True),
            _scorecard("r2", 4.5, True),
            _scorecard("r3", 4.5, True),
        ]
    })
    out2 = run_e1_human_qa_panel(
        cands2, c2_client=c2b, scorecard_client=scb, min_reviewers=3, min_battles=10
    )
    v2 = out2["/tmp/i2.ckpt"]
    assert v2.verdict == "inconclusive"
    assert v2.stop_condition_met is False
    assert v2.n_battles == 5  # coverage axis below the min


def test_c2_deploy_invoked_per_candidate():
    """(6) The fake c2_client records the candidate path deployed; assert the
    deploy is invoked once per non-None candidate path."""
    cands = E1CandidateSet(
        post_d_path="/tmp/a.ckpt",
        post_c3_best_path="/tmp/b.ckpt",
        post_b_path="/tmp/c.ckpt",
    )
    c2 = _FakeC2Client(battle_count_per_series=10)
    sc = _FakeScorecardClient({
        "/tmp/a.ckpt": [_scorecard("r1", 4.0, True), _scorecard("r2", 4.0, True), _scorecard("r3", 4.0, True)],
        "/tmp/b.ckpt": [_scorecard("r1", 3.5, False), _scorecard("r2", 3.5, False), _scorecard("r3", 3.5, False)],
        "/tmp/c.ckpt": [_scorecard("r1", 2.0, False), _scorecard("r2", 2.0, False), _scorecard("r3", 2.0, False)],
    })
    run_e1_human_qa_panel(
        cands, c2_client=c2, scorecard_client=sc, min_reviewers=3, min_battles=10
    )
    assert c2.deployed_paths == ["/tmp/a.ckpt", "/tmp/b.ckpt", "/tmp/c.ckpt"]
    assert c2.start_calls == 3  # one start_series per candidate


def test_verdict_map_keys_are_candidate_paths():
    """(7) The returned dict is keyed by candidate_path; iterate E1CandidateSet
    post-D first, post-C3, post-B; Nones dropped; dedup."""
    # Dedup: post_d and post_c3_best_path point at the SAME path.
    cands = E1CandidateSet(
        post_d_path="/tmp/same.ckpt",
        post_c3_best_path="/tmp/same.ckpt",  # dedup with post_d
        post_b_path="/tmp/other.ckpt",
    )
    c2 = _FakeC2Client(battle_count_per_series=10)
    sc = _FakeScorecardClient({
        "/tmp/same.ckpt": [_scorecard("r1", 4.0, True), _scorecard("r2", 4.0, True), _scorecard("r3", 4.0, True)],
        "/tmp/other.ckpt": [_scorecard("r1", 3.0, False), _scorecard("r2", 3.0, False), _scorecard("r3", 3.0, False)],
    })
    out = run_e1_human_qa_panel(
        cands, c2_client=c2, scorecard_client=sc, min_reviewers=3, min_battles=10
    )
    # Deduped to 2 unique paths; keys are candidate paths.
    assert set(out.keys()) == {"/tmp/same.ckpt", "/tmp/other.ckpt"}
    # Nones dropped: a candidate set with all-Nones yields an empty map.
    empty_cands = E1CandidateSet()
    out_empty = run_e1_human_qa_panel(
        empty_cands, c2_client=c2, scorecard_client=sc, min_reviewers=3, min_battles=10
    )
    assert out_empty == {}
    # Ordering: post-D first, post-C3, post-B (list() preserves insertion).
    cands2 = E1CandidateSet(
        post_d_path="/tmp/d.ckpt",
        post_c3_best_path="/tmp/c3.ckpt",
        post_b_path="/tmp/b.ckpt",
    )
    sc2 = _FakeScorecardClient({
        "/tmp/d.ckpt": [_scorecard("r1", 4.0, True), _scorecard("r2", 4.0, True), _scorecard("r3", 4.0, True)],
        "/tmp/c3.ckpt": [_scorecard("r1", 3.5, False), _scorecard("r2", 3.5, False), _scorecard("r3", 3.5, False)],
        "/tmp/b.ckpt": [_scorecard("r1", 2.5, False), _scorecard("r2", 2.5, False), _scorecard("r3", 2.5, False)],
    })
    out2 = run_e1_human_qa_panel(
        cands2, c2_client=_FakeC2Client(battle_count_per_series=10),
        scorecard_client=sc2, min_reviewers=3, min_battles=10,
    )
    assert list(out2.keys()) == ["/tmp/d.ckpt", "/tmp/c3.ckpt", "/tmp/b.ckpt"]


def test_json_scorecard_client_roundtrip(tmp_path):
    """(8) JsonScorecardClient reads a scorecards JSON file, aggregates into a
    HumanQAVerdict per candidate; round-trips a write+read."""
    json_path = tmp_path / "scorecards.json"
    client = JsonScorecardClient(str(json_path))
    # Write two scorecards for cand A, one for cand B (list-of-rows form).
    client.submit_scorecard("/tmp/a.ckpt", reviewer_id="r1", difficulty_score=4.5, harder_than_baseline=True, notes="tough")
    client.submit_scorecard("/tmp/a.ckpt", reviewer_id="r2", difficulty_score=4.0, harder_than_baseline=True, notes="")
    client.submit_scorecard("/tmp/b.ckpt", reviewer_id="r1", difficulty_score=2.5, harder_than_baseline=False, notes="easy")
    # Read back.
    rows_a = client.list_scorecards("/tmp/a.ckpt")
    rows_b = client.list_scorecards("/tmp/b.ckpt")
    assert len(rows_a) == 2
    assert len(rows_b) == 1
    # Aggregate into a verdict.
    v_a = aggregate_scorecards("/tmp/a.ckpt", rows_a, n_battles=10, min_reviewers=2, min_battles=10)
    assert v_a.verdict == "harder"
    assert v_a.mean_difficulty_score == pytest.approx((4.5 + 4.0) / 2)
    assert v_a.n_harder_than_baseline == 2
    assert v_a.stop_condition_met is True
    assert "tough" in v_a.freeform_notes
    # Also accept the dict-keyed JSON form on read.
    dict_form_path = tmp_path / "scorecards_dict.json"
    dict_form_path.write_text(json.dumps({
        "/tmp/a.ckpt": [
            {"reviewer_id": "r1", "difficulty_score": 4.5, "harder_than_baseline": True, "notes": "tough"},
            {"reviewer_id": "r2", "difficulty_score": 4.0, "harder_than_baseline": True, "notes": ""},
        ],
    }))
    client2 = JsonScorecardClient(str(dict_form_path))
    rows_a2 = client2.list_scorecards("/tmp/a.ckpt")
    assert len(rows_a2) == 2
    assert all(r.get("candidate_path") == "/tmp/a.ckpt" for r in rows_a2)


def test_mean_difficulty_aggregation():
    """(9) mean_difficulty_score = mean of difficulty_score across reviewers
    (exact); n_harder_than_baseline = count where harder_than_baseline True."""
    rows = [
        _scorecard("r1", 4.0, True),
        _scorecard("r2", 3.0, False),
        _scorecard("r3", 5.0, True),
        _scorecard("r4", 2.0, False),
    ]
    v = aggregate_scorecards("/tmp/m.ckpt", rows, n_battles=10, min_reviewers=4, min_battles=10)
    assert v.mean_difficulty_score == pytest.approx((4.0 + 3.0 + 5.0 + 2.0) / 4)
    assert v.n_harder_than_baseline == 2
    assert v.n_reviewers == 4
    # Empty scorecards -> mean 0.0, n_harder 0, inconclusive (n_reviewers=0 < min).
    v_empty = aggregate_scorecards("/tmp/m0.ckpt", [], n_battles=10, min_reviewers=1, min_battles=10)
    assert v_empty.mean_difficulty_score == 0.0
    assert v_empty.n_harder_than_baseline == 0
    assert v_empty.verdict == "inconclusive"
    assert v_empty.stop_condition_met is False


def test_mana_draw_blind_scope_documented():
    """(10) The panel evaluates a mana_draw-BLIND V5 bot (V5RlhfAdapter
    discards mana_draw_logit) -- the mana_draw axis is NOT exercised by the
    panel (exercised only by the prod _get_action_v5 path wired in E5). A
    module-level constant documents this; assert it is present + non-empty."""
    assert isinstance(MANA_DRAW_BLIND_SCOPE, str)
    assert "mana_draw-BLIND" in MANA_DRAW_BLIND_SCOPE
    assert "v5_rlhf_adapter.py:201" in MANA_DRAW_BLIND_SCOPE
    # The module docstring also documents the mana_draw-BLIND scope.
    import train_v3.e1_human_qa_panel as mod
    assert "mana_draw-BLIND" in mod.__doc__
    # E4 does NOT construct V5RlhfAdapter for the C2 path: assert the module
    # does NOT import V5RlhfAdapter (the C2 path uses the checkpoint path +
    # factory _factory_v5_real resolves kind=v5; E4 only constructs it if
    # standalone inference is needed, which it is not). The module docstring
    # MAY reference the adapter name in prose (documenting the mana_draw-BLIND
    # scope); the regression guard is on the IMPORT, not the docstring text.
    import inspect
    src = inspect.getsource(mod)
    assert "import V5RlhfAdapter" not in src, (
        "E4 must NOT import V5RlhfAdapter for the C2 path "
        "(C2 uses the checkpoint path + factory resolves kind=v5)."
    )
    assert "from rlhf_env.components.v5_rlhf_adapter" not in src, (
        "E4 must NOT import from v5_rlhf_adapter (C2 uses the checkpoint path "
        "+ factory resolves kind=v5)."
    )
    # The module namespace must NOT bind V5RlhfAdapter.
    assert not hasattr(mod, "V5RlhfAdapter"), "E4 must NOT bind V5RlhfAdapter."


def test_comparable_verdict():
    """(11) mean in [3.0, 4.0) -> 'comparable'."""
    v, stop = derive_verdict(
        mean_difficulty_score=3.5,
        n_harder_than_baseline=0,
        n_reviewers=3,
        n_battles=10,
        min_reviewers=3,
        min_battles=10,
    )
    assert v == "comparable"
    assert stop is True
    # boundary: 3.0 is comparable (inclusive lower), 4.0 is harder (exclusive upper)
    v_lo, _ = derive_verdict(
        mean_difficulty_score=3.0, n_harder_than_baseline=0,
        n_reviewers=3, n_battles=10, min_reviewers=3, min_battles=10,
    )
    assert v_lo == "comparable"
    v_hi, _ = derive_verdict(
        mean_difficulty_score=4.0, n_harder_than_baseline=0,
        n_reviewers=3, n_battles=10, min_reviewers=3, min_battles=10,
    )
    assert v_hi == "harder"


def test_regression_guard_no_edit_to_composed():
    """(12) e1_human_qa_panel.py imports C2CollectionDriver from
    c2_collection_driver (does NOT redefine it) and does NOT add an MCP tool /
    import rlhf_env.mcp_server."""
    import inspect
    import train_v3.e1_human_qa_panel as mod
    src = inspect.getsource(mod)
    # Imports C2CollectionDriver -- does NOT redefine it.
    assert "from rlhf_env.components.c2_collection_driver import" in src
    assert "class C2CollectionDriver" not in src, "E4 must NOT redefine C2CollectionDriver"
    # Does NOT add an MCP tool / import rlhf_env.mcp_server.
    assert "rlhf_env.mcp_server" not in src, "E4 must NOT import rlhf_env.mcp_server"
    assert "rlhf_env.components.c2_collection_driver" in src  # READ-ONLY composition
    assert "train_v3.c_to_d_handoff" in src  # READ-ONLY E1CandidateSet
    # The imported C2CollectionDriver is the SAME class (identity, not a copy).
    from rlhf_env.components.c2_collection_driver import C2CollectionDriver as Orig
    from train_v3.e1_human_qa_panel import C2CollectionDriver as Imp
    assert Orig is Imp


def test_panel_driver_class_run_matches_function():
    """Sanity: E1HumanQAPanelDriver.run and run_e1_human_qa_panel agree."""
    cands = E1CandidateSet(post_d_path="/tmp/x.ckpt")
    c2 = _FakeC2Client(battle_count_per_series=10)
    sc = _FakeScorecardClient({
        "/tmp/x.ckpt": [_scorecard("r1", 4.0, True), _scorecard("r2", 4.0, True), _scorecard("r3", 4.0, True)]
    })
    drv = E1HumanQAPanelDriver(min_reviewers=3, min_battles=10)
    out_drv = drv.run(cands, c2_client=c2, scorecard_client=sc)
    c2b = _FakeC2Client(battle_count_per_series=10)
    out_fn = run_e1_human_qa_panel(cands, c2_client=c2b, scorecard_client=sc, min_reviewers=3, min_battles=10)
    assert out_drv["/tmp/x.ckpt"].verdict == out_fn["/tmp/x.ckpt"].verdict == "harder"


def test_panel_does_not_raise_on_any_verdict():
    """The SOFT-gate property: the panel returns the verdict map for every
    verdict class without raising (harder / comparable / easier / inconclusive
    all come back as ordinary return values)."""
    for score, harder_flag, expected in [
        (4.5, True, "harder"),
        (3.5, False, "comparable"),
        (2.0, False, "easier"),
    ]:
        cands = E1CandidateSet(post_d_path=f"/tmp/{expected}.ckpt")
        c2 = _FakeC2Client(battle_count_per_series=10)
        sc = _FakeScorecardClient({
            f"/tmp/{expected}.ckpt": [
                _scorecard("r1", score, harder_flag),
                _scorecard("r2", score, harder_flag),
                _scorecard("r3", score, harder_flag),
            ]
        })
        out = run_e1_human_qa_panel(
            cands, c2_client=c2, scorecard_client=sc, min_reviewers=3, min_battles=10
        )
        assert out[f"/tmp/{expected}.ckpt"].verdict == expected