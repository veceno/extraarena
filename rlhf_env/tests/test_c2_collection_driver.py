from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rlhf_env.components.c2_collection_driver import C2CollectionDriver


def _draw(source: str = "human") -> dict[str, Any]:
    return {"action_type": "mana_draw", "decision_source": source}


class FakeObserverClient:
    def __init__(self, groups: list[dict[str, Any]], plans: dict[str, dict[str, Any]]):
        self.groups = groups
        self.plans = plans
        self.start_series_called = False

    def start_series(self, spec):  # pragma: no cover - safety sentinel
        self.start_series_called = True
        raise AssertionError("observer must not create human matches through MCP")

    def list_v5_groups(self, *args, **kwargs):
        assert kwargs["battle_tag"] == "human-vs-rl"
        return {"groups": self.groups}

    def get_v5_dataset_summary(self, group_id: str):
        return self.plans[group_id]["summary"]

    def validate_v5_traces(self, group_id: str):
        return self.plans[group_id].get("validation", {"checked": 1, "ok": 1, "broken": []})

    def get_v5_trace(self, group_id: str, battle_id: str, what: str):
        return {"data": self.plans[group_id][what].get(battle_id, {})}


def _plan(tmp_path: Path, gid: str, weights_hash: str, *, actions=None, warnings=None, valid=True):
    bid = f"b_{gid}"
    return {
        "summary": {
            "group_id": gid,
            "group_dir": str(tmp_path / gid),
            "battles_finished": 1,
            "battle_ids": [bid],
            "battles": [{"battle_id": bid, "policy_warnings": warnings or [], "degraded": bool(warnings)}],
            "current_card_count": 50,
            "current_catalog_hash": "catalog50",
        },
        "validation": {"checked": 1, "ok": 1 if valid else 0, "broken": [] if valid else [{"battle_id": bid}]},
        "meta": {bid: {
            "battle_tag": "human-vs-rl", "p1_actor_type": "human",
            "bot_policy": {"kind": "v5", "weights_hash": weights_hash},
            "catalog_hash": "catalog50",
        }},
        "actions": {bid: actions or []},
    }


def _driver(tmp_path: Path, **kwargs):
    checkpoint = tmp_path / "u29250.onnx"
    checkpoint.write_bytes(b"real-v5-weights")
    return C2CollectionDriver(
        str(checkpoint), expected_catalog_hash="catalog50", **kwargs,
    ), hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16]


def test_observes_completed_groups_without_starting_human_match(tmp_path):
    driver, whash = _driver(tmp_path, mana_draw_floor=99, battle_cap=10)
    client = FakeObserverClient(
        [{"group_id": "g1", "finished_at": "now"}],
        {"g1": _plan(tmp_path, "g1", whash, actions=[_draw(), _draw(), _draw("bot")])},
    )
    result = driver.collect(client)
    assert result.status == "ok"
    assert result.group_ids == ["g1"]
    assert result.group_dirs == [str(tmp_path / "g1")]
    assert result.battle_count == 1
    assert result.mana_draw_row_count == 2
    assert client.start_series_called is False


def test_waits_for_human_data_and_ignores_in_progress(tmp_path):
    driver, whash = _driver(tmp_path)
    client = FakeObserverClient(
        [{"group_id": "running", "finished_at": None}],
        {"running": _plan(tmp_path, "running", whash)},
    )
    result = driver.collect(client)
    assert result.status == "skipped"
    assert result.stopped_reason == "waiting_for_human_data"


def test_rejects_degraded_stale_or_structurally_broken_groups(tmp_path):
    driver, whash = _driver(tmp_path, mana_draw_floor=99, battle_cap=10)
    plans = {
        "fallback": _plan(tmp_path, "fallback", whash, warnings=["fallback"]),
        "stale": _plan(tmp_path, "stale", "wrong-hash"),
        "broken": _plan(tmp_path, "broken", whash, valid=False),
        "clean": _plan(tmp_path, "clean", whash, actions=[_draw()]),
    }
    groups = [{"group_id": gid, "finished_at": "now"} for gid in plans]
    result = driver.collect(FakeObserverClient(groups, plans))
    assert result.group_ids == ["clean"]
    assert {r["group_id"] for r in result.rejected_groups} == {"fallback", "stale", "broken"}


def test_consumed_groups_are_not_replayed_twice(tmp_path):
    driver, whash = _driver(tmp_path, mana_draw_floor=99, battle_cap=10)
    client = FakeObserverClient(
        [{"group_id": "g1", "finished_at": "now"}],
        {"g1": _plan(tmp_path, "g1", whash, actions=[_draw()])},
    )
    assert driver.collect(client).status == "ok"
    assert driver.collect(client).status == "skipped"


def test_floor_cap_and_series_plan_shape(tmp_path):
    driver, whash = _driver(tmp_path, mana_draw_floor=2, battle_cap=2500, battles_per_series=1000)
    plans = {
        gid: _plan(tmp_path, gid, whash, actions=[_draw(), _draw()])
        for gid in ("g1", "g2")
    }
    groups = [{"group_id": gid, "finished_at": "now"} for gid in plans]
    result = driver.collect(FakeObserverClient(groups, plans))
    assert result.stopped_reason == "floor"
    assert result.group_ids == ["g1"]
    specs = driver.plan_series_specs()
    assert [s["battles_planned"] for s in specs] == [1000, 1000, 500]
    assert all(s["p1_actor_type"] == "human" and s["p2_model"]["kind"] == "v5" for s in specs)
