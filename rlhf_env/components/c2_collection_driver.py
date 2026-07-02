"""C2 collection driver — drives V5-vs-human series in rlhf_env and harvests
V5 training traces until the D-C4 mana_draw floor or the D-C8 battle cap is hit.

Block C component C1 (D-C0: build in the worktree's rlhf_env now).

C2 (design.md:125): "deploy current best V5 vs humans in rlhf_env" and collect
the resulting V5 traces. This driver orchestrates that collection:

  * ``plan_series_spec(start_index)`` — builds a start_series spec with
    ``p1_actor_type='human'`` + ``p2_model={name:'v5-deploy', path, kind:'v5'}``
    (``kind='v5'`` BYPASSES ``_sidecar_kind_detector`` which does not recognize
    V5, ``policy_adapters.py:213-242``) + ``battles_planned`` capped at 1000 per
    spec (``mcp_server.py:167`` maximum=1000). Multiple series are planned to
    reach ``battle_cap`` (``battle_cap // battles_per_series`` series, remainder
    in the last series).

  * ``collect(mcp_client)`` — drives series via ``mcp_client.start_series(spec)``
    and iterates ``next_battle`` to completion; for each finished group it polls
    ``list_v5_groups`` + ``get_v5_dataset_summary`` + reads the v5_trace
    ``actions`` rows (``get_v5_trace(what='actions')``); counts mana_draw rows =
    rows with ``action_type=='mana_draw'`` AND ``decision_source=='human'``
    (D-C4 NEW counter — ``get_v5_dataset_summary`` does NOT count these, so the
    driver counts them itself from the action rows; v5_trace emits
    action_type+decision_source, ``v5_trace.py:456``). REJECTS groups where
    policy_fallbacks fired (``mcp_server.py:82``) OR ``v5_trace_ok`` is false
    (avoid stub/garbage traces). STOPS when ``mana_draw_row_count >=
    mana_draw_floor`` OR ``finished_battle_count >= battle_cap`` (whichever
    first, D-C4).

SYNTHETIC-testable: ``mcp_client`` is an injected fake (a Protocol with
``start_series`` / ``next_battle`` / ``list_v5_groups`` / ``get_v5_dataset_summary``
/ ``get_v5_trace`` returning canned data); NO real rlhf_env/DB/socket/onnx.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# mcp_server.py:167 — battles_planned maximum per series spec.
_BATTLES_PER_SPEC_CAP = 1000


class McpCollectionClient(Protocol):
    """Minimal MCP client surface the driver needs (injectable fake for tests).

    ``list_battles`` is the PRIMARY battle-id source against a real server: the
    real ``get_v5_dataset_summary`` (mcp_server.py:716-724) returns NO battle_ids,
    so per-battle action-row enumeration must come from a battle-listing tool.
    A real MCP tool exposing the manifest's ``battles_results`` battle_ids
    (mcp_server.py:703-708) is the intended backing implementation; until that
    tool exists, a real client's ``list_battles`` may raise NotImplementedError,
    which the driver treats as "no battle_ids resolvable" (and the group's
    mana_draw count stays 0). The summary 'battle_ids' field remains a
    test-injection override.
    """

    def start_series(self, spec: Dict[str, Any]) -> Dict[str, Any]: ...
    def next_battle(self, group_id: str) -> Dict[str, Any]: ...
    def list_v5_groups(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: ...
    def get_v5_dataset_summary(self, group_id: str) -> Dict[str, Any]: ...
    def get_v5_trace(self, group_id: str, battle_id: str, what: str) -> Dict[str, Any]: ...
    def list_battles(self, group_id: str) -> List[Dict[str, Any]]: ...


@dataclass
class C2CollectionResult:
    mana_draw_row_count: int = 0
    battle_count: int = 0
    groups_collected: int = 0
    stopped_reason: str = ""  # 'floor' | 'cap' | 'no_more_series'
    rejected_groups: List[Dict[str, Any]] = field(default_factory=list)
    series_plans: List[Dict[str, Any]] = field(default_factory=list)


class C2CollectionDriver:
    """Drive V5-vs-human series and harvest V5 traces (D-C4 mana_draw floor +
    D-C8 battle cap)."""

    def __init__(
        self,
        v5_checkpoint_path: str,
        *,
        mana_draw_floor: int = 5000,
        battle_cap: int = 5000,
        battles_per_series: int = 1000,
    ) -> None:
        if battles_per_series <= 0:
            raise ValueError("battles_per_series must be > 0")
        self.v5_checkpoint_path = v5_checkpoint_path
        self.mana_draw_floor = int(mana_draw_floor)
        self.battle_cap = int(battle_cap)
        self.battles_per_series = int(battles_per_series)

    # ------------------------------------------------------------------
    # spec planning
    # ------------------------------------------------------------------

    def plan_series_specs(self) -> List[Dict[str, Any]]:
        """Plan the full list of series specs to reach ``battle_cap`` battles.

        Each spec carries at most ``min(battles_per_series, _BATTLES_PER_SPEC_CAP)``
        battles; the final series absorbs the remainder.
        """
        per_spec = min(self.battles_per_series, _BATTLES_PER_SPEC_CAP)
        if per_spec <= 0:
            raise ValueError("effective battles_per_series must be > 0")
        specs: List[Dict[str, Any]] = []
        produced = 0
        idx = 0
        while produced < self.battle_cap:
            remaining = self.battle_cap - produced
            count = min(per_spec, remaining)
            specs.append(self.plan_series_spec(idx, battles_planned=count))
            produced += count
            idx += 1
        return specs

    def plan_series_spec(self, start_index: int = 0, *, battles_planned: Optional[int] = None) -> Dict[str, Any]:
        """A start_series spec: human p1 vs deployed V5 p2.

        ``p1_actor_type='human'`` (browser human, decision_source='human' in
        V5 traces) + ``p2_model`` object with ``kind='v5'`` (bypasses
        ``_sidecar_kind_detector`` which does not recognize V5).
        """
        count = battles_planned if battles_planned is not None else self.battles_per_series
        count = min(int(count), _BATTLES_PER_SPEC_CAP)
        count = max(count, 1)
        return {
            "p1_actor_type": "human",
            "p2_model": {
                "name": "v5-deploy",
                "path": self.v5_checkpoint_path,
                "kind": "v5",
            },
            "battles_planned": count,
            "starting_player": "random",
            "seed": 1000 + int(start_index),
        }

    # ------------------------------------------------------------------
    # collection
    # ------------------------------------------------------------------

    def collect(self, mcp_client: McpCollectionClient) -> C2CollectionResult:
        """Drive series via ``mcp_client`` and harvest V5 traces.

        Stops when mana_draw_row_count >= mana_draw_floor (D-C4 'floor') OR
        finished_battle_count >= battle_cap (D-C8 'cap'), whichever first.
        """
        result = C2CollectionResult()
        plans = self.plan_series_specs()
        result.series_plans = list(plans)

        for spec in plans:
            if result.mana_draw_row_count >= self.mana_draw_floor:
                result.stopped_reason = "floor"
                return result
            if result.battle_count >= self.battle_cap:
                result.stopped_reason = "cap"
                return result

            start_resp = mcp_client.start_series(spec)
            group_id = start_resp.get("group_id") if isinstance(start_resp, dict) else None
            # policy_fallbacks in the start response → reject the whole group.
            fallbacks = self._policy_fallbacks(start_resp)
            trace_ok = self._v5_trace_ok(start_resp)

            # Drive the series to completion via next_battle.
            battles_in_series = 0
            if group_id is not None:
                while True:
                    if result.battle_count >= self.battle_cap:
                        break
                    nb = mcp_client.next_battle(group_id)
                    status = nb.get("status") if isinstance(nb, dict) else None
                    if status == "series_complete":
                        break
                    # A new battle was started; accumulate fallbacks from the
                    # per-battle response too.
                    fallbacks.extend(self._policy_fallbacks(nb))
                    battles_in_series += 1

            # Harvest the group's V5 traces.
            group_mana_draw, group_battles, group_trace_ok = self._harvest_group(
                mcp_client, group_id
            )
            # A group is REJECTED if any fallback fired OR v5_trace_ok is false.
            group_rejected = bool(fallbacks) or (not group_trace_ok) or (not trace_ok)
            if group_rejected:
                result.rejected_groups.append({
                    "group_id": group_id,
                    "policy_fallbacks": fallbacks,
                    "v5_trace_ok": group_trace_ok and trace_ok,
                    "battles_in_series": battles_in_series,
                })
                logger.warning(
                    "[c2] rejected group %s: fallbacks=%s trace_ok=%s",
                    group_id, fallbacks, group_trace_ok and trace_ok,
                )
                continue

            result.mana_draw_row_count += group_mana_draw
            result.battle_count += group_battles
            result.groups_collected += 1

        if result.mana_draw_row_count >= self.mana_draw_floor:
            result.stopped_reason = "floor"
        elif result.battle_count >= self.battle_cap:
            result.stopped_reason = "cap"
        else:
            result.stopped_reason = "no_more_series"
        return result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _policy_fallbacks(resp: Any) -> List[str]:
        if not isinstance(resp, dict):
            return []
        # start_series response: 'policy_warnings' (mcp_server.py:422);
        # next_battle response: 'policy_warnings' (mcp_server.py:440).
        w = resp.get("policy_warnings") or resp.get("policy_fallbacks") or []
        return list(w) if isinstance(w, (list, tuple)) else []

    @staticmethod
    def _v5_trace_ok(resp: Any) -> bool:
        # A start_series/next_battle response does not carry v5_trace_ok for the
        # whole group; default True and let _harvest_group compute the real value.
        if not isinstance(resp, dict):
            return True
        v = resp.get("v5_trace_ok")
        return True if v is None else bool(v)

    def _harvest_group(
        self, mcp_client: McpCollectionClient, group_id: Optional[str]
    ) -> (int, int, bool):
        """Count mana_draw human rows + finished battles + v5_trace_ok for a group.

        Returns (mana_draw_row_count, finished_battle_count, v5_trace_ok).
        """
        if group_id is None:
            return 0, 0, False
        summary = mcp_client.get_v5_dataset_summary(group_id)
        if not isinstance(summary, dict) or summary.get("error"):
            return 0, 0, False
        finished = int(summary.get("battles_finished", 0) or 0)
        v5_ok = int(summary.get("v5_trace_ok_count", 0) or 0)
        # Group trace_ok only if EVERY finished battle has a valid trace.
        group_trace_ok = (finished > 0) and (v5_ok >= finished)

        mana_draw_rows = 0
        # Enumerate finished battles battle-by-battle and read each battle's
        # actions rows. The real get_v5_dataset_summary returns NO battle_ids
        # (mcp_server.py:716-724), so _battle_ids falls back to
        # mcp_client.list_battles(group_id) — the PRIMARY real-server source
        # (a battle-listing MCP tool backed by the manifest's battles_results,
        # mcp_server.py:703-708). The summary 'battle_ids' field is a
        # test-injection override. We count rows with action_type=='mana_draw'
        # AND decision_source=='human' (D-C4 NEW counter).
        battle_ids = self._battle_ids(summary, mcp_client, group_id)
        for bid in battle_ids:
            trace = mcp_client.get_v5_trace(group_id, bid, "actions")
            rows = trace.get("data", []) if isinstance(trace, dict) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if (row.get("action_type") == "mana_draw"
                        and row.get("decision_source") == "human"):
                    mana_draw_rows += 1
        return mana_draw_rows, finished, group_trace_ok

    @staticmethod
    def _battle_ids(summary: Dict[str, Any], mcp_client: McpCollectionClient, group_id: str) -> List[str]:
        """Resolve the list of finished battle_ids for a group.

        Source precedence (so the D-C4 mana_draw counter is NOT inert against a
        real server — the real ``get_v5_dataset_summary`` returns no battle_ids):
          1. explicit ``battle_ids`` field on the summary (test-injection
             override / future summary extension);
          2. ``battles`` list of ``{battle_id}`` dicts on the summary;
          3. ``mcp_client.list_battles(group_id)`` — the PRIMARY real-server
             source (a battle-listing MCP tool backed by the manifest's
             ``battles_results``, mcp_server.py:703-708). Missing/raising ->
             no battle_ids resolvable.
        """
        ids = summary.get("battle_ids")
        if isinstance(ids, list):
            return [str(b) for b in ids]
        battles = summary.get("battles")
        if isinstance(battles, list):
            out = []
            for b in battles:
                if isinstance(b, dict) and b.get("battle_id"):
                    out.append(str(b["battle_id"]))
                elif isinstance(b, str):
                    out.append(b)
            return out
        # Primary real-server source: ask the client to list the group's battles.
        list_battles = getattr(mcp_client, "list_battles", None)
        if callable(list_battles):
            try:
                battles = list_battles(group_id)
            except NotImplementedError:
                return []
            if isinstance(battles, list):
                out = []
                for b in battles:
                    if isinstance(b, dict) and b.get("battle_id"):
                        out.append(str(b["battle_id"]))
                    elif isinstance(b, str):
                        out.append(b)
                return out
        return []