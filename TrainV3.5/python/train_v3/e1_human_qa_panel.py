"""E4 -- human-QA panel driver (Block E1, E-E8 SOFT gate).

NEW TrainV3.5 component COMPOSING (READ-ONLY) the C2 deploy surface
(``rlhf_env.components.c2_collection_driver.C2CollectionDriver`` +
``McpCollectionClient`` + ``C2CollectionResult``) and ``c_to_d_handoff.E1CandidateSet``
to deploy each E1 candidate checkpoint vs humans in rlhf_env and capture
post-battle reviewer scorecards (subjective-difficulty Likert 1-5).

ZERO infrastructure exists today for the subjective-difficulty capture -- E4
authors it.

The panel is a SOFT gate (E-E8): a "harder" verdict is a release-checklist item
+ a soft pass; an "easier"/"inconclusive" verdict is a soft warn (NOT a hard
fail -- the hard ship decision is the E3 threshold table + the E5
export-parity/fallback-guard). E4 NEVER raises/blocks on a verdict: it emits
the verdict map and returns.

mana_draw-BLIND SCOPE (load-bearing, E-E8): the rlhf_env deploy adapter
``V5RlhfAdapter.select_action`` (``v5_rlhf_adapter.py:127-212``) handles the V5
3-tuple forward but DISCARDS ``mana_draw_logit`` (``:201`` underscore-bound,
argmaxes ONLY over the 601 logits, ``:205-209``; ``mana_draw_legal_mask`` /
``select_includes_mana_draw`` are NEVER referenced in ``v5_rlhf_adapter.py``).
The panel therefore evaluates a V5 bot that NEVER takes mana_draw actions --
the "subjectively harder" verdict must be interpreted accordingly (the
mana_draw axis is exercised only by the prod ``_get_action_v5`` path wired in
E5, NOT by the panel). E4 reuses the adapter UNCHANGED; E4 does NOT construct
``V5RlhfAdapter`` for the C2 path (C2 uses the checkpoint ``path`` + the
factory ``_factory_v5_real`` resolves ``kind='v5'``).

NO edit to ``c2_collection_driver.py`` / ``v5_rlhf_adapter.py`` /
``c_to_d_handoff.py`` / ``e1_tournament.py`` / ``rlhf_env/mcp_server.py`` -- E4
is a NEW sibling composing them. E4 does NOT add an MCP tool in
``rlhf_env/mcp_server.py``; it defines a ``ReviewerScorecardClient`` Protocol
(sibling to C2 ``McpCollectionClient``, injectable fake-able) + a
``JsonScorecardClient`` (JSON-file ingestion, the USER-run operational entry)
-- mirrors C2 Protocol-injection.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

# C2 deploy surface -- READ-ONLY composition. The import path
# ``rlhf_env.components.c2_collection_driver`` resolves via the worktree root
# being on sys.path (the test bootstrap inserts it; the operational USER-run
# entry sets PYTHONPATH=worktree-root:TrainV3.5/python).
from rlhf_env.components.c2_collection_driver import (  # noqa: E402
    C2CollectionDriver,
    C2CollectionResult,
    McpCollectionClient,
)

from train_v3.c_to_d_handoff import E1CandidateSet  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# mana_draw-BLIND SCOPE marker -- the panel evaluates a V5 bot that NEVER takes
# mana_draw actions (V5RlhfAdapter.select_action discards mana_draw_logit,
# v5_rlhf_adapter.py:201 / :205-209). The mana_draw axis is exercised only by
# the prod _get_action_v5 path wired in E5. Surfaced as a module constant so a
# regression test can assert it (test_mana_draw_blind_scope_documented).
# ---------------------------------------------------------------------------
MANA_DRAW_BLIND_SCOPE: str = (
    "E4 panel evaluates a mana_draw-BLIND V5 bot: V5RlhfAdapter.select_action "
    "discards mana_draw_logit (v5_rlhf_adapter.py:201, :205-209); the panel "
    "never exercises the mana_draw axis (that axis is exercised only by the "
    "prod _get_action_v5 path wired in E5)."
)

__all__ = [
    "HumanQAVerdict",
    "ReviewerScorecardClient",
    "JsonScorecardClient",
    "E1HumanQAPanelDriver",
    "run_e1_human_qa_panel",
    "MANA_DRAW_BLIND_SCOPE",
    "derive_verdict",
    "aggregate_scorecards",
]

# A mana_draw_floor sentinel that NEVER trips the C2 floor check. The spec
# (BLOCK_E1_PLAN.md E4) says "E4 sets C2 mana_draw_floor=0 (never trips floor)"
# -- but C2's floor check is ``mana_draw_row_count >= mana_draw_floor``
# (``c2_collection_driver.py:158``), so ``floor=0`` trips IMMEDIATELY (``0 >= 0``
# is True -> the driver returns with ``stopped_reason='floor'`` and
# ``battle_count=0``). The load-bearing INTENT is "never trips floor"; the
# cleanest faithful realization is a huge sentinel (effectively infinity) so
# the ``>=`` check is never satisfied. E4's stop is the SUBJECTIVE-COVERAGE
# objective (``n_reviewers >= min_reviewers`` AND ``n_battles >= min_battles``),
# NOT the C2 training-data floor.
_NEVER_FLOOR = 10**18


# =============================================================================
# HumanQAVerdict -- the subjective-difficulty verdict schema (frozen dataclass)
# =============================================================================
@dataclass(frozen=True)
class HumanQAVerdict:
    """The subjective-difficulty verdict for one E1 candidate (E-E8 SOFT gate).

    Fields:
      candidate_path: the E1 candidate checkpoint path the verdict applies to.
      n_battles: the number of battles deployed vs humans (coverage axis).
      n_reviewers: the number of reviewer scorecards collected (coverage axis).
      mean_difficulty_score: mean of the per-reviewer ``difficulty_score`` (1-5
        Likert, "subjectively harder for humans"). 0.0 when no scorecards.
      n_harder_than_baseline: count of reviewers marking the candidate harder
        than the current prod baseline ``extra-lr-v4-max``.
      verdict: 'harder' / 'comparable' / 'easier' / 'inconclusive'. Derived from
        ``mean_difficulty_score`` + ``n_harder_than_baseline`` thresholds;
        'inconclusive' when coverage (``n_reviewers`` / ``n_battles``) is below
        the panel minimum. Exposed as a top-level attribute (``.verdict``) for
        E3 duck-type interop (``e1_tournament.py:566`` reads
        ``meta.get('human_qa_verdict')``; the E3 test fake
        ``_HumanQAFake`` at ``tests/test_e1_tournament.py:130-136`` relies on a
        ``.verdict`` attribute -- this dataclass matches that surface).
      freeform_notes: aggregated reviewer freeform notes (joined with '; '),
        default "" when no notes were submitted.
      stop_condition_met: True iff ``n_reviewers >= min_reviewers`` AND
        ``n_battles >= min_battles`` (the subjective-coverage stop; NOT the C2
        mana_draw-floor/battle-cap, which is a training-data objective).
    """

    candidate_path: str
    n_battles: int
    n_reviewers: int
    mean_difficulty_score: float
    n_harder_than_baseline: int
    verdict: str
    freeform_notes: str = ""
    stop_condition_met: bool = False


# =============================================================================
# ReviewerScorecardClient -- the injectable fake-able scorecard surface
# (Protocol, sibling to C2 McpCollectionClient)
# =============================================================================
class ReviewerScorecardClient(Protocol):
    """Minimal scorecard-client surface the panel needs (injectable fake for
    tests, sibling to C2 ``McpCollectionClient``).

    A reviewer submits one scorecard row per candidate via ``submit_scorecard``;
    the panel harvests the collected rows via ``list_scorecards``. The
    operational backing implementation is ``JsonScorecardClient`` (a JSON file
    the USER populates); tests inject a fake implementing this Protocol.
    """

    def submit_scorecard(
        self,
        candidate_path: str,
        *,
        reviewer_id: str,
        difficulty_score: float,
        harder_than_baseline: bool,
        notes: str = "",
    ) -> dict:
        """Record one reviewer scorecard row for a candidate. Returns the row
        dict that was recorded."""
        ...

    def list_scorecards(self, candidate_path: str) -> List[dict]:
        """Return the collected scorecard rows for a candidate (each row is a
        dict with keys ``candidate_path`` / ``reviewer_id`` /
        ``difficulty_score`` / ``harder_than_baseline`` / ``notes``)."""
        ...


# =============================================================================
# JsonScorecardClient -- the USER-run operational entry (implements
# ReviewerScorecardClient; reads/writes a scorecards JSON file)
# =============================================================================
class JsonScorecardClient:
    """JSON-file-backed ``ReviewerScorecardClient`` -- the USER-run operational
    entry. Reads/writes a scorecards JSON file.

    The JSON schema is EITHER:
      (a) a list of scorecard dicts: ``[{candidate_path, reviewer_id,
          difficulty_score, harder_than_baseline, notes}, ...]``; OR
      (b) a dict keyed by ``candidate_path``: ``{candidate_path:
          [{reviewer_id, difficulty_score, harder_than_baseline, notes}, ...]}``.

    On read, both forms are normalized to the list-of-rows form (each row
    carries ``candidate_path``). On write (``submit_scorecard``), the file is
    rewritten in the list-of-rows form (canonical). Round-trips a write+read.
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)

    # -- read ---------------------------------------------------------------
    def _load_rows(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return self._normalize(data)

    @staticmethod
    def _normalize(data: Any) -> List[dict]:
        rows: List[dict] = []
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    rows.append(dict(row))
            return rows
        if isinstance(data, dict):
            # dict keyed by candidate_path -> rows
            for cand_path, entries in data.items():
                if isinstance(entries, list):
                    for row in entries:
                        if isinstance(row, dict):
                            # stamp candidate_path if missing
                            r = dict(row)
                            r.setdefault("candidate_path", cand_path)
                            rows.append(r)
            return rows
        return rows

    # -- write --------------------------------------------------------------
    def _dump_rows(self, rows: List[dict]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, sort_keys=True)

    # -- ReviewerScorecardClient surface ------------------------------------
    def submit_scorecard(
        self,
        candidate_path: str,
        *,
        reviewer_id: str,
        difficulty_score: float,
        harder_than_baseline: bool,
        notes: str = "",
    ) -> dict:
        row = {
            "candidate_path": candidate_path,
            "reviewer_id": reviewer_id,
            "difficulty_score": float(difficulty_score),
            "harder_than_baseline": bool(harder_than_baseline),
            "notes": str(notes),
        }
        rows = self._load_rows()
        rows.append(row)
        self._dump_rows(rows)
        return row

    def list_scorecards(self, candidate_path: str) -> List[dict]:
        return [r for r in self._load_rows() if r.get("candidate_path") == candidate_path]


# =============================================================================
# Verdict derivation + scorecard aggregation (pure helpers, unit-testable)
# =============================================================================
def derive_verdict(
    *,
    mean_difficulty_score: float,
    n_harder_than_baseline: int,
    n_reviewers: int,
    n_battles: int,
    min_reviewers: int,
    min_battles: int,
) -> tuple:
    """Derive the (verdict, stop_condition_met) tuple from coverage + scores.

    Precedence (E-E8):
      1. 'inconclusive' -- coverage not met (``n_reviewers < min_reviewers`` OR
         ``n_battles < min_battles``); ``stop_condition_met`` False.
      2. 'harder' -- ``mean_difficulty_score >= 4.0`` OR
         ``n_harder_than_baseline >= ceil(n_reviewers / 2)``.
      3. 'comparable' -- ``mean_difficulty_score`` in ``[3.0, 4.0)``.
      4. 'easier' -- ``mean_difficulty_score < 3.0``.
    """
    stop_met = (n_reviewers >= min_reviewers) and (n_battles >= min_battles)
    if not stop_met:
        return "inconclusive", False
    if mean_difficulty_score >= 4.0 or n_harder_than_baseline >= math.ceil(n_reviewers / 2):
        return "harder", True
    if mean_difficulty_score >= 3.0:
        return "comparable", True
    return "easier", True


def aggregate_scorecards(
    candidate_path: str,
    scorecards: List[dict],
    *,
    n_battles: int,
    min_reviewers: int,
    min_battles: int,
) -> HumanQAVerdict:
    """Aggregate the per-reviewer scorecard rows for one candidate into a
    ``HumanQAVerdict``.

    ``mean_difficulty_score`` = mean of ``difficulty_score`` across reviewers
    (0.0 when no scorecards). ``n_harder_than_baseline`` = count where
    ``harder_than_baseline`` is truthy. ``freeform_notes`` = the non-empty
    reviewer ``notes`` joined with '; ' (default "").
    """
    n_reviewers = len(scorecards)
    if n_reviewers > 0:
        mean_score = sum(float(r.get("difficulty_score", 0.0)) for r in scorecards) / n_reviewers
    else:
        mean_score = 0.0
    n_harder = sum(1 for r in scorecards if bool(r.get("harder_than_baseline", False)))
    notes = "; ".join(str(r.get("notes", "")) for r in scorecards if str(r.get("notes", "")).strip())
    verdict, stop_met = derive_verdict(
        mean_difficulty_score=mean_score,
        n_harder_than_baseline=n_harder,
        n_reviewers=n_reviewers,
        n_battles=n_battles,
        min_reviewers=min_reviewers,
        min_battles=min_battles,
    )
    return HumanQAVerdict(
        candidate_path=candidate_path,
        n_battles=n_battles,
        n_reviewers=n_reviewers,
        mean_difficulty_score=mean_score,
        n_harder_than_baseline=n_harder,
        verdict=verdict,
        freeform_notes=notes,
        stop_condition_met=stop_met,
    )


def _iter_candidate_paths(candidates: E1CandidateSet) -> List[str]:
    """Iterate the E1 candidate paths: post-D first, post-C3, post-B; drop
    Nones; dedup (preserve first-occurrence order)."""
    seen: set = set()
    out: List[str] = []
    for p in (candidates.post_d_path, candidates.post_c3_best_path, candidates.post_b_path):
        if p is None:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


# =============================================================================
# E1HumanQAPanelDriver -- composes C2CollectionDriver (READ-ONLY) + the
# scorecard client to produce a HumanQAVerdict per candidate
# =============================================================================
class E1HumanQAPanelDriver:
    """Drive V5-vs-human deploys per E1 candidate (via ``C2CollectionDriver``,
    READ-ONLY) + harvest reviewer scorecards (via a
    ``ReviewerScorecardClient``) + aggregate into a ``HumanQAVerdict`` per
    candidate.

    The stop condition is SUBJECTIVE-COVERAGE: ``n_reviewers >= min_reviewers``
    AND ``n_battles >= min_battles`` (NOT C2's mana_draw-floor/battle-cap,
    which is a training-data objective). The driver sets C2
    ``mana_draw_floor=_NEVER_FLOOR`` (a huge sentinel -- ``floor=0`` would trip
    C2's ``>=`` check immediately; the load-bearing intent is "never trips
    floor") + ``battle_cap=min_battles`` (the coverage floor); the scorecard
    client supplies the reviewer-coverage axis between series.
    """

    def __init__(
        self,
        *,
        min_reviewers: int,
        min_battles: int,
        battles_per_series: int = 1000,
    ) -> None:
        if min_reviewers < 0:
            raise ValueError("min_reviewers must be >= 0")
        if min_battles < 0:
            raise ValueError("min_battles must be >= 0")
        self.min_reviewers = int(min_reviewers)
        self.min_battles = int(min_battles)
        self.battles_per_series = int(battles_per_series)

    def run(
        self,
        candidates: E1CandidateSet,
        *,
        c2_client: McpCollectionClient,
        scorecard_client: ReviewerScorecardClient,
    ) -> Dict[str, HumanQAVerdict]:
        """Deploy each candidate vs humans + collect reviewer scorecards +
        emit the verdict map keyed by ``candidate_path``.

        SOFT gate (E-E8): NEVER raises/blocks on a verdict. A 'harder' verdict
        is a release-checklist item + soft pass; an 'easier'/'inconclusive'
        verdict is a soft warn (NOT a hard fail).
        """
        verdicts: Dict[str, HumanQAVerdict] = {}
        for path in _iter_candidate_paths(candidates):
            # Deploy the candidate vs humans via C2 (READ-ONLY composition).
            # mana_draw_floor=_NEVER_FLOOR -> never trips the C2 floor (E4 stop
            # is a subjective-coverage objective, NOT a training-data objective;
            # see the _NEVER_FLOOR comment for why floor=0 would trip
            # immediately). battle_cap=min_battles -> the coverage floor.
            driver = C2CollectionDriver(
                path,
                mana_draw_floor=_NEVER_FLOOR,
                battle_cap=self.min_battles,
                battles_per_series=self.battles_per_series,
            )
            # Optional operational hook: arrange/show the browser-owned human
            # series for this candidate. C2 itself remains a disk observer and
            # never creates a human match in a separate MCP process.
            prepare_candidate = getattr(c2_client, "prepare_candidate", None)
            if callable(prepare_candidate):
                prepare_candidate(path, driver.plan_series_specs())
            result: C2CollectionResult = driver.collect(c2_client)
            # Harvest the reviewer scorecards for this candidate.
            scorecards = scorecard_client.list_scorecards(path)
            verdict = aggregate_scorecards(
                path,
                scorecards,
                n_battles=result.battle_count,
                min_reviewers=self.min_reviewers,
                min_battles=self.min_battles,
            )
            verdicts[path] = verdict
            logger.info(
                "[e1-qa] candidate=%s verdict=%s n_battles=%d n_reviewers=%d mean=%.3f n_harder=%d stop=%s",
                path,
                verdict.verdict,
                verdict.n_battles,
                verdict.n_reviewers,
                verdict.mean_difficulty_score,
                verdict.n_harder_than_baseline,
                verdict.stop_condition_met,
            )
        return verdicts


# =============================================================================
# run_e1_human_qa_panel -- the USER-run operational entry
# =============================================================================
def run_e1_human_qa_panel(
    candidates: E1CandidateSet,
    *,
    c2_client: McpCollectionClient,
    scorecard_client: ReviewerScorecardClient,
    min_reviewers: int,
    min_battles: int,
    battles_per_series: int = 1000,
) -> Dict[str, HumanQAVerdict]:
    """USER-run operational entry: deploy each E1 candidate vs humans via the
    C2 driver + collect reviewer scorecards + emit the verdict map keyed by
    ``candidate_path``.

    Iterates ``E1CandidateSet`` paths: post-D first, post-C3, post-B; Nones
    dropped; dedup. The verdict is a SOFT gate (E-E8): a 'harder' verdict is a
    release-checklist item + a soft pass; an 'easier'/'inconclusive' verdict is
    a soft warn (NOT a hard fail -- the hard ship decision is the E3 threshold
    table + the E5 export-parity/fallback-guard).
    """
    driver = E1HumanQAPanelDriver(
        min_reviewers=min_reviewers,
        min_battles=min_battles,
        battles_per_series=battles_per_series,
    )
    return driver.run(candidates, c2_client=c2_client, scorecard_client=scorecard_client)
