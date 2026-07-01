"""C1 — C2CollectionDriver unit tests (synthetic: fake MCP client + canned
v5_trace catalog; no real rlhf_env/DB/socket/onnx).

Covers:
  5. counts mana_draw rows per D-C4 (mana_draw + decision_source=='human').
  6. stops on mana_draw floor (stopped_reason='floor').
  7. stops on battle cap (stopped_reason='cap').
  8. rejects fallback/stub traces; multi-series 1000-cap handling
     (battle_cap=5000 -> 5 series planned); p1_actor_type='human' + kind='v5'
     in the planned spec.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from rlhf_env.components.c2_collection_driver import (
    C2CollectionDriver,
    C2CollectionResult,
)


# ----------------------------------------------------------------------------
# Fake MCP client (canned catalog)
# ----------------------------------------------------------------------------

class _GroupPlan:
    def __init__(
        self,
        *,
        battles_finished: int = 1,
        v5_trace_ok_count: int | None = None,
        actions_rows: List[Dict[str, Any]] | None = None,
        policy_warnings: List[str] | None = None,
        battle_ids: List[str] | None = None,
    ):
        self.battles_finished = battles_finished
        self.v5_trace_ok_count = (
            v5_trace_ok_count if v5_trace_ok_count is not None else battles_finished
        )
        self.actions_rows = actions_rows or []
        self.policy_warnings = policy_warnings or []
        # One battle_id per finished battle by default.
        self.battle_ids = battle_ids or [f"b_{i}" for i in range(battles_finished)]


class FakeMcpClient:
    """Injectable fake implementing the McpCollectionClient Protocol."""

    def __init__(self, group_plans: List[_GroupPlan]):
        self._plans = list(group_plans)
        self._idx = 0
        self._current: _GroupPlan | None = None
        self.start_series_calls: List[Dict[str, Any]] = []

    def start_series(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        self.start_series_calls.append(spec)
        if self._idx >= len(self._plans):
            return {"group_id": None, "policy_warnings": []}
        plan = self._plans[self._idx]
        self._idx += 1
        self._current = plan
        return {
            "group_id": f"g_{self._idx - 1}",
            "policy_warnings": list(plan.policy_warnings),
        }

    def next_battle(self, group_id: str) -> Dict[str, Any]:
        # Immediately signal series completion; battle counts come from summary.
        return {"status": "series_complete"}

    def list_v5_groups(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"groups": []}

    def get_v5_dataset_summary(self, group_id: str) -> Dict[str, Any]:
        plan = self._current
        if plan is None:
            return {"error": "no group"}
        return {
            "battles_finished": plan.battles_finished,
            "v5_trace_ok_count": plan.v5_trace_ok_count,
            "battle_ids": list(plan.battle_ids),
        }

    def get_v5_trace(self, group_id: str, battle_id: str, what: str) -> Dict[str, Any]:
        plan = self._current
        if plan is None or what != "actions":
            return {"data": []}
        # actions_rows is the PER-GROUP catalog; return it once for the first
        # battle_id and empty for the rest (so per-group counts are not inflated
        # by multiplying across battle_ids).
        if not plan.battle_ids or battle_id != plan.battle_ids[0]:
            return {"data": []}
        return {"data": list(plan.actions_rows)}


def _mana_draw_row(decision_source: str = "human") -> Dict[str, Any]:
    return {"action_type": "mana_draw", "decision_source": decision_source}


def _row(atype: str, decision_source: str = "human") -> Dict[str, Any]:
    return {"action_type": atype, "decision_source": decision_source}


# ----------------------------------------------------------------------------
# 5. mana_draw row counting (D-C4)
# ----------------------------------------------------------------------------

def test_mana_draw_row_counting_d_c4():
    rows = [
        _mana_draw_row("human"),  # count
        _mana_draw_row("human"),  # count
        _mana_draw_row("human"),  # count
        _mana_draw_row("human"),  # count
        _mana_draw_row("bot"),    # skip (not human)
        _row("end_turn", "human"),  # skip (not mana_draw)
        _row("play_card", "human"),  # skip
        _mana_draw_row("llm"),    # skip (not human)
    ]
    plan = _GroupPlan(battles_finished=1, actions_rows=rows)
    client = FakeMcpClient([plan])
    driver = C2CollectionDriver("/v5.onnx", mana_draw_floor=10_000, battle_cap=1, battles_per_series=1)
    result = driver.collect(client)
    assert result.mana_draw_row_count == 4
    assert result.battle_count == 1
    assert result.groups_collected == 1


# ----------------------------------------------------------------------------
# 6. stops on mana_draw floor
# ----------------------------------------------------------------------------

def test_stops_on_mana_draw_floor():
    # 5 series planned (battle_cap=5000, per_series=1000); each group 1000
    # battles + 1000 mana_draw human rows; floor=3000 -> floor hit after 3
    # groups, before the cap (battle_count=3000 < 5000).
    plans = [
        _GroupPlan(
            battles_finished=1000,
            actions_rows=[_mana_draw_row("human") for _ in range(1000)],
        )
        for _ in range(5)
    ]
    client = FakeMcpClient(plans)
    driver = C2CollectionDriver(
        "/v5.onnx", mana_draw_floor=3000, battle_cap=5000, battles_per_series=1000
    )
    result = driver.collect(client)
    assert result.stopped_reason == "floor"
    assert result.mana_draw_row_count >= 3000
    assert result.battle_count < 5000


# ----------------------------------------------------------------------------
# 7. stops on battle cap
# ----------------------------------------------------------------------------

def test_stops_on_battle_cap():
    # 3 series planned (battle_cap=3000, per_series=1000); each group 1000
    # battles + 100 mana_draw rows; floor=5000 -> cap hit after 3 groups
    # (battle_count=3000), mana_draw=300 < 5000.
    plans = [
        _GroupPlan(
            battles_finished=1000,
            actions_rows=[_mana_draw_row("human") for _ in range(100)],
        )
        for _ in range(3)
    ]
    client = FakeMcpClient(plans)
    driver = C2CollectionDriver(
        "/v5.onnx", mana_draw_floor=5000, battle_cap=3000, battles_per_series=1000
    )
    result = driver.collect(client)
    assert result.stopped_reason == "cap"
    assert result.battle_count >= 3000
    assert result.mana_draw_row_count < 5000


# ----------------------------------------------------------------------------
# 8. rejects fallback/stub traces + multi-series planning + spec shape
# ----------------------------------------------------------------------------

def test_rejects_fallback_and_stub_trace_groups():
    # group0: policy_fallbacks fired -> rejected.
    # group1: v5_trace_ok false (0 < battles_finished) -> rejected.
    # group2: clean -> accepted.
    plans = [
        _GroupPlan(
            battles_finished=1,
            actions_rows=[_mana_draw_row("human")],
            policy_warnings=["v5-stub -> end_turn (fallback)"],
        ),
        _GroupPlan(
            battles_finished=2,
            v5_trace_ok_count=0,
            actions_rows=[_mana_draw_row("human")],
        ),
        _GroupPlan(
            battles_finished=1,
            actions_rows=[_mana_draw_row("human")],
        ),
    ]
    client = FakeMcpClient(plans)
    driver = C2CollectionDriver(
        "/v5.onnx", mana_draw_floor=10_000, battle_cap=3, battles_per_series=1
    )
    result = driver.collect(client)
    assert len(result.rejected_groups) == 2
    rejected_gids = {r["group_id"] for r in result.rejected_groups}
    assert "g_0" in rejected_gids  # fallback group
    assert "g_1" in rejected_gids  # stub-trace group
    # Only the clean group contributed.
    assert result.groups_collected == 1
    assert result.battle_count == 1
    assert result.mana_draw_row_count == 1


def test_multi_series_planning_and_spec_shape():
    driver = C2CollectionDriver(
        "/v5/checkpoint.onnx", battle_cap=5000, battles_per_series=1000
    )
    specs = driver.plan_series_specs()
    assert len(specs) == 5
    total = 0
    for spec in specs:
        assert spec["p1_actor_type"] == "human"
        p2 = spec["p2_model"]
        assert isinstance(p2, dict)
        assert p2["kind"] == "v5"
        assert p2["name"] == "v5-deploy"
        assert p2["path"] == "/v5/checkpoint.onnx"
        assert spec["battles_planned"] <= 1000
        total += spec["battles_planned"]
    assert total == 5000


def test_plan_series_spec_remainder_handling():
    # battle_cap=2500, per_series=1000 -> 3 series (1000, 1000, 500).
    driver = C2CollectionDriver("/v5.onnx", battle_cap=2500, battles_per_series=1000)
    specs = driver.plan_series_specs()
    assert [s["battles_planned"] for s in specs] == [1000, 1000, 500]


def test_plan_series_spec_single_shape():
    driver = C2CollectionDriver("/v5/x.onnx", battle_cap=1, battles_per_series=1000)
    spec = driver.plan_series_spec(0, battles_planned=1)
    assert spec["p1_actor_type"] == "human"
    assert spec["p2_model"]["kind"] == "v5"
    assert spec["battles_planned"] == 1


# ----------------------------------------------------------------------------
# 9. D-C4 counter is NOT inert against a real server: a summary WITHOUT an
#    injected 'battle_ids' field still yields the correct mana_draw count when
#    the client provides list_battles (the PRIMARY real-server battle-id source;
#    the real get_v5_dataset_summary returns no battle_ids, mcp_server.py:716).
# ----------------------------------------------------------------------------

class _ListBattlesFakeMcpClient(FakeMcpClient):
    """Like FakeMcpClient but does NOT inject 'battle_ids' into the summary;
    instead it exposes battle_ids via list_battles (the real-server path)."""

    def get_v5_dataset_summary(self, group_id: str) -> Dict[str, Any]:
        plan = self._current
        if plan is None:
            return {"error": "no group"}
        # Deliberately omit 'battle_ids' to mimic the real server summary.
        return {
            "battles_finished": plan.battles_finished,
            "v5_trace_ok_count": plan.v5_trace_ok_count,
        }

    def list_battles(self, group_id: str) -> List[Dict[str, Any]]:
        plan = self._current
        if plan is None:
            return []
        return [{"battle_id": bid} for bid in plan.battle_ids]


def test_mana_draw_count_via_list_battles_real_server_path():
    """A summary without 'battle_ids' (real-server shape) still produces the
    correct mana_draw count via mcp_client.list_battles."""
    rows = [
        _mana_draw_row("human"),
        _mana_draw_row("human"),
        _row("play_card", "human"),
        _mana_draw_row("bot"),
    ]
    plan = _GroupPlan(battles_finished=2, actions_rows=rows)
    client = _ListBattlesFakeMcpClient([plan])
    driver = C2CollectionDriver("/v5.onnx", mana_draw_floor=10_000, battle_cap=2, battles_per_series=2)
    result = driver.collect(client)
    assert result.mana_draw_row_count == 2
    assert result.battle_count == 2
    assert result.groups_collected == 1


def test_list_battles_not_implemented_yields_zero_no_crash():
    """If a real client's list_battles is not yet wired (NotImplementedError),
    the driver does not crash and the group's mana_draw count stays 0."""

    class _NoListBattlesClient(FakeMcpClient):
        def get_v5_dataset_summary(self, group_id: str) -> Dict[str, Any]:
            plan = self._current
            if plan is None:
                return {"error": "no group"}
            return {
                "battles_finished": plan.battles_finished,
                "v5_trace_ok_count": plan.v5_trace_ok_count,
            }

        def list_battles(self, group_id: str) -> List[Dict[str, Any]]:
            raise NotImplementedError("battle-listing MCP tool not yet deployed")

    rows = [_mana_draw_row("human"), _mana_draw_row("human")]
    plan = _GroupPlan(battles_finished=1, actions_rows=rows)
    client = _NoListBattlesClient([plan])
    driver = C2CollectionDriver("/v5.onnx", mana_draw_floor=10_000, battle_cap=1, battles_per_series=1)
    result = driver.collect(client)
    # No battle_ids resolvable -> mana_draw_rows stays 0; group still accepted
    # (trace_ok True) but contributes 0 mana_draw rows and the finished battles.
    # battle_count (1) reaches battle_cap (1) -> 'cap'.
    assert result.mana_draw_row_count == 0
    assert result.battle_count == 1
    assert result.groups_collected == 1
    assert result.stopped_reason == "cap"